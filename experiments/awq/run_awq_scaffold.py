import argparse
import platform
import sys
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.awq.algorithm import (
    build_calibration_batch,
    evaluate_openpangu_perplexity_sequential,
    quantize_openpangu_awq_sequential,
)
from experiments.common.benchmark import apply_benchmark_overrides, evaluate_multiple_choice, load_benchmark_plan
from experiments.common.config import load_config, resolve_path
from experiments.common.data import load_text_samples
from experiments.common.inventory import collect_linear_inventory, select_target_modules, validate_target_modules
from experiments.common.metrics import measure_generation
from experiments.common.reporting import create_run_dir, summarize_directory, write_json
from experiments.common.runtime import ensure_hf_home, select_runtime
from experiments.compressed_artifacts.io import export_quant_artifact


def load_awq_runtime_model(model_path: Path, hf_home: Optional[Path]):
    hf_home = ensure_hf_home(hf_home)
    runtime_device, model_dtype = select_runtime()
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        use_fast=False,
        trust_remote_code=True,
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        torch_dtype=model_dtype,
        local_files_only=True,
    )
    model.eval()
    return tokenizer, model, runtime_device, model_dtype


def measure_generation_single_device(model, tokenizer, prompts, system_prompt, max_new_tokens, runtime_device):
    if runtime_device.type != "cuda":
        return {"skipped": True, "reason": "generation benchmark requires cuda runtime"}

    try:
        torch.cuda.reset_peak_memory_stats(runtime_device)
        model.to(runtime_device)
        model.eval()
        result = measure_generation(
            model,
            tokenizer,
            prompts,
            system_prompt,
            max_new_tokens,
            runtime_device,
            "",
        )
        result["peak_memory_mb"] = round(torch.cuda.max_memory_allocated(runtime_device) / (1024 * 1024), 2)
        return result
    except RuntimeError as exc:
        return {"skipped": True, "reason": str(exc)}
    finally:
        model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def evaluate_benchmark_single_device(model, tokenizer, benchmark_plan, system_prompt, runtime_device):
    if runtime_device.type != "cuda":
        return {"skipped": True, "reason": "benchmark evaluation requires cuda runtime", "results": []}

    try:
        model.to(runtime_device)
        model.eval()
        return evaluate_multiple_choice(
            model,
            tokenizer,
            benchmark_plan["samples"],
            system_prompt,
            runtime_device,
            "",
            benchmark_plan["max_length"],
            benchmark_plan["scoring_mode"],
        )
    except RuntimeError as exc:
        return {"skipped": True, "reason": str(exc), "results": []}
    finally:
        model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser(description="AWQ scaffold for OpenPangu.")
    parser.add_argument("--config", required=True, help="Path to experiment JSON config.")
    parser.add_argument("--model-path", default="", help="Optional override for config.model_path.")
    parser.add_argument("--hf-home", default="", help="Optional override for config.hf_home.")
    parser.add_argument("--output-dir", default="", help="Optional override for config.output_dir.")
    parser.add_argument("--save-dir", default="", help="Optional override for config.awq.save_dir.")
    parser.add_argument("--experiment-name-suffix", default="", help="Optional suffix appended to experiment_name.")
    parser.add_argument("--min-layer", type=int, default=None, help="Optional override for awq.min_layer.")
    parser.add_argument("--max-layer", type=int, default=None, help="Optional override for awq.max_layer.")
    parser.add_argument("--bits", type=int, default=None, help="Optional override for awq.bits.")
    parser.add_argument("--group-size", type=int, default=None, help="Optional override for awq.group_size.")
    parser.add_argument("--calibration-limit", type=int, default=None, help="Optional override for calibration_data.limit.")
    parser.add_argument(
        "--calibration-max-length",
        type=int,
        default=None,
        help="Optional override for calibration_data.max_length and evaluation.perplexity_max_length.",
    )
    parser.add_argument("--benchmark-data", default="", help="Optional override for config.benchmark_data.path.")
    parser.add_argument("--benchmark-limit", type=int, default=None, help="Optional override for config.benchmark_data.limit.")
    parser.add_argument("--benchmark-max-length", type=int, default=None, help="Optional override for config.benchmark_evaluation.max_length.")
    parser.add_argument(
        "--benchmark-scoring-mode",
        default="",
        help='Optional override for config.benchmark_evaluation.scoring_mode. Use "avg_logprob" or "total_logprob".',
    )
    args = parser.parse_args()

    config, _ = load_config(args.config)
    if args.model_path:
        config["model_path"] = args.model_path
    if args.hf_home:
        config["hf_home"] = args.hf_home
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.save_dir:
        config["awq"]["save_dir"] = args.save_dir
    if args.experiment_name_suffix:
        config["experiment_name"] = f"{config['experiment_name']}_{args.experiment_name_suffix}"
    if args.min_layer is not None:
        config["awq"]["min_layer"] = args.min_layer
    if args.max_layer is not None:
        config["awq"]["max_layer"] = args.max_layer
    if args.bits is not None:
        config["awq"]["bits"] = args.bits
    if args.group_size is not None:
        config["awq"]["group_size"] = args.group_size
    if args.calibration_limit is not None:
        config["calibration_data"]["limit"] = args.calibration_limit
    if args.calibration_max_length is not None:
        config["calibration_data"]["max_length"] = args.calibration_max_length
        config["evaluation"]["perplexity_max_length"] = args.calibration_max_length
    apply_benchmark_overrides(
        config,
        benchmark_data=args.benchmark_data,
        benchmark_limit=args.benchmark_limit,
        benchmark_max_length=args.benchmark_max_length,
        benchmark_scoring_mode=args.benchmark_scoring_mode,
    )

    model_path = resolve_path(REPO_ROOT, config["model_path"])
    hf_home = resolve_path(REPO_ROOT, config.get("hf_home"))
    output_root = resolve_path(REPO_ROOT, config["output_dir"])
    calibration_path = resolve_path(REPO_ROOT, config["calibration_data"].get("path"))
    evaluation_data_cfg = config.get("evaluation_data", {})
    evaluation_path = resolve_path(REPO_ROOT, evaluation_data_cfg.get("path"))

    run_dir = create_run_dir(output_root, "awq", config["experiment_name"])
    write_json(run_dir / "config_snapshot.json", config)

    tokenizer, model, runtime_device, model_dtype = load_awq_runtime_model(model_path, hf_home)
    inventory = collect_linear_inventory(model)
    targets = select_target_modules(
        inventory,
        config["module_selection"].get("include_groups", []),
        config["module_selection"].get("exclude_patterns", []),
    )
    validate_target_modules(
        inventory,
        targets,
        config["module_selection"].get("include_groups", []),
        config["module_selection"].get("exclude_patterns", []),
    )
    texts = load_text_samples(calibration_path, config["calibration_data"]["limit"])
    evaluation_texts = load_text_samples(
        evaluation_path if evaluation_data_cfg else calibration_path,
        evaluation_data_cfg.get("limit", config["calibration_data"]["limit"]),
    )
    benchmark_plan = load_benchmark_plan(REPO_ROOT, config)

    summary = {
        "method": "awq",
        "mode": config["awq"]["mode"],
        "runtime": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "runtime_device": str(runtime_device),
            "torch_dtype": str(model_dtype),
            "device_map": "",
        },
        "calibration_samples": len(texts),
        "module_inventory": {
            "total_linear_modules": inventory["total_linear_modules"],
            "total_linear_parameters": inventory["total_linear_parameters"],
            "group_counts": inventory["group_counts"],
            "target_module_count": len(targets),
        },
        "awq_plan": {
            "bits": config["awq"]["bits"],
            "group_size": config["awq"]["group_size"],
            "alpha_grid": config["awq"]["alpha_grid"],
            "sym": config["awq"]["sym"],
        },
        "source_validation": {
            "paper_reference": "Lin et al. 2024 AWQ",
            "expected_upstream_snapshot": str((REPO_ROOT.parent / "external" / "llm-awq").resolve()),
            "scaffold_reuse": [
                "experiments.common.config",
                "experiments.common.data",
                "experiments.common.inventory",
                "experiments.common.reporting",
            ],
        },
    }
    if benchmark_plan is not None:
        summary["benchmark_plan"] = {
            "path": str(benchmark_plan["path"]),
            "task_slug": benchmark_plan["task_slug"],
            "sample_count": benchmark_plan["limit"],
            "max_length": benchmark_plan["max_length"],
            "scoring_mode": benchmark_plan["scoring_mode"],
            "apply_chat_template": benchmark_plan.get("apply_chat_template", True),
            "prompt_style": benchmark_plan.get("prompt_style"),
            "prompt_template_source": benchmark_plan.get("prompt_template_source"),
            "few_shot_count": benchmark_plan.get("few_shot_count"),
        }

    calibration_batch = build_calibration_batch(
        tokenizer,
        texts,
        config["system_prompt"],
        int(config["calibration_data"].get("max_length", config["evaluation"]["perplexity_max_length"])),
        apply_chat_template=not bool(config["calibration_data"].get("raw_text", False)),
    )
    summary["calibration_batch"] = {
        "sequence_length": calibration_batch["sequence_length"],
        "sample_count": int(calibration_batch["input_ids"].shape[0]),
    }
    evaluation_batch = build_calibration_batch(
        tokenizer,
        evaluation_texts,
        config["system_prompt"],
        int(evaluation_data_cfg.get("max_length", config["evaluation"]["perplexity_max_length"])),
        apply_chat_template=not bool(evaluation_data_cfg.get("raw_text", False)),
    )
    summary["evaluation_batch"] = {
        "sequence_length": evaluation_batch["sequence_length"],
        "sample_count": int(evaluation_batch["input_ids"].shape[0]),
    }

    if config["evaluation"]["run_perplexity"]:
        summary["baseline_perplexity"] = evaluate_openpangu_perplexity_sequential(
            model,
            evaluation_batch["input_ids"],
            evaluation_batch["attention_mask"],
            runtime_device,
        )
    if config["evaluation"].get("run_generation", False):
        summary["baseline_generation"] = measure_generation_single_device(
            model,
            tokenizer,
            evaluation_texts[: config["evaluation"]["generation_samples"]],
            config["system_prompt"],
            config["evaluation"]["max_new_tokens"],
            runtime_device,
        )
    if benchmark_plan is not None:
        baseline_benchmark = evaluate_benchmark_single_device(
            model,
            tokenizer,
            benchmark_plan,
            config["system_prompt"],
            runtime_device,
        )
        summary["baseline_benchmark"] = {
            "sample_count": baseline_benchmark["sample_count"],
            "evaluated_count": baseline_benchmark["evaluated_count"],
            "skipped_count": baseline_benchmark["skipped_count"],
            "correct_count": baseline_benchmark["correct_count"],
            "accuracy": baseline_benchmark["accuracy"],
            "scoring_mode": baseline_benchmark["scoring_mode"],
            "tasks": baseline_benchmark["tasks"],
        }
        write_json(run_dir / "baseline_benchmark_predictions.json", baseline_benchmark["results"])

    quant_stats = quantize_openpangu_awq_sequential(
        model,
        calibration_batch["input_ids"],
        calibration_batch["attention_mask"],
        runtime_device,
        {module["name"] for module in targets},
        config["awq"],
    )
    artifact_payloads = quant_stats.pop("artifact_payloads", {})
    summary["awq_stats"] = {
        "module_count": quant_stats["module_count"],
        "total_quantized_params": quant_stats["total_quantized_params"],
        "total_target_params": quant_stats["total_target_params"],
        "quantized_fraction": quant_stats["quantized_fraction"],
        "layer_count": len(quant_stats["layers"]),
        "elapsed_s": quant_stats["elapsed_s"],
        "peak_memory_mb": quant_stats["peak_memory_mb"],
    }

    if config["evaluation"].get("run_post_quant_perplexity", True):
        summary["quantized_perplexity"] = evaluate_openpangu_perplexity_sequential(
            model,
            evaluation_batch["input_ids"],
            evaluation_batch["attention_mask"],
            runtime_device,
        )
    if config["evaluation"].get("run_generation", False):
        summary["quantized_generation"] = measure_generation_single_device(
            model,
            tokenizer,
            evaluation_texts[: config["evaluation"]["generation_samples"]],
            config["system_prompt"],
            config["evaluation"]["max_new_tokens"],
            runtime_device,
        )
    if benchmark_plan is not None:
        quantized_benchmark = evaluate_benchmark_single_device(
            model,
            tokenizer,
            benchmark_plan,
            config["system_prompt"],
            runtime_device,
        )
        summary["quantized_benchmark"] = {
            "sample_count": quantized_benchmark["sample_count"],
            "evaluated_count": quantized_benchmark["evaluated_count"],
            "skipped_count": quantized_benchmark["skipped_count"],
            "correct_count": quantized_benchmark["correct_count"],
            "accuracy": quantized_benchmark["accuracy"],
            "scoring_mode": quantized_benchmark["scoring_mode"],
            "tasks": quantized_benchmark["tasks"],
        }
        write_json(run_dir / "quantized_benchmark_predictions.json", quantized_benchmark.get("results", []))

    save_dir = resolve_path(run_dir, config["awq"].get("save_dir"))
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        summary["saved_model_dir"] = str(save_dir)
        summary["saved_model_format"] = "huggingface_dense_checkpoint_with_dequantized_rewritten_weights"
        summary["saved_model_info"] = summarize_directory(save_dir)

    compressed_artifact_dir = resolve_path(run_dir, config["awq"].get("compressed_artifact_dir"))
    if compressed_artifact_dir is not None:
        compressed_info = export_quant_artifact(
            artifact_payloads,
            compressed_artifact_dir,
            method="awq",
            base_model_path_hint=str(model_path),
            source_model_path=str(save_dir) if save_dir is not None else str(model_path),
        )
        summary["compressed_artifact_dir"] = str(compressed_artifact_dir)
        summary["compressed_artifact_format"] = "openpangu_quant_overlay_packed_nbit_v1"
        summary["compressed_artifact_info"] = compressed_info

    write_json(run_dir / "linear_inventory.json", inventory)
    write_json(run_dir / "target_modules.json", targets)
    write_json(run_dir / "awq_stats.json", quant_stats)
    write_json(run_dir / "summary.json", summary)

    print(f"[OK] run_dir={run_dir}")
    print(f"[OK] target_module_count={len(targets)}")
    if "baseline_perplexity" in summary:
        print(f"[OK] baseline_perplexity={summary['baseline_perplexity']}")
    if "quantized_perplexity" in summary:
        print(f"[OK] quantized_perplexity={summary['quantized_perplexity']}")
    if "baseline_generation" in summary and not summary["baseline_generation"].get("skipped"):
        print(f"[OK] baseline_generation_tokens_per_second={summary['baseline_generation']['tokens_per_second']}")
    if "quantized_generation" in summary and not summary["quantized_generation"].get("skipped"):
        print(f"[OK] quantized_generation_tokens_per_second={summary['quantized_generation']['tokens_per_second']}")
    if "baseline_benchmark" in summary:
        print(f"[OK] baseline_benchmark_accuracy={summary['baseline_benchmark']['accuracy']}")
    if "quantized_benchmark" in summary:
        print(f"[OK] quantized_benchmark_accuracy={summary['quantized_benchmark']['accuracy']}")
    print("[OK] AWQ sequential quantization port executed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
