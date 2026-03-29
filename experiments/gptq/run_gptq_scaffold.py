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

from experiments.common.config import load_config, resolve_path
from experiments.common.data import load_text_samples
from experiments.common.inventory import collect_linear_inventory, select_target_modules
from experiments.common.metrics import measure_generation, measure_perplexity
from experiments.common.reporting import create_run_dir, write_json
from experiments.common.runtime import ensure_hf_home, load_tokenizer_and_model, select_runtime
from experiments.gptq.algorithm import (
    build_calibration_batch,
    evaluate_openpangu_perplexity_sequential,
    quantize_openpangu_sequential,
)


def load_gptq_runtime_model(model_path: Path, hf_home: Optional[Path]):
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


def main() -> int:
    parser = argparse.ArgumentParser(description="GPTQ scaffold for OpenPangu.")
    parser.add_argument("--config", required=True, help="Path to experiment JSON config.")
    parser.add_argument("--model-path", default="", help="Optional override for config.model_path.")
    parser.add_argument("--hf-home", default="", help="Optional override for config.hf_home.")
    parser.add_argument("--output-dir", default="", help="Optional override for config.output_dir.")
    parser.add_argument("--save-dir", default="", help="Optional override for config.gptq.save_dir.")
    parser.add_argument("--experiment-name-suffix", default="", help="Optional suffix appended to experiment_name.")
    parser.add_argument("--min-layer", type=int, default=None, help="Optional override for gptq.min_layer.")
    parser.add_argument("--max-layer", type=int, default=None, help="Optional override for gptq.max_layer.")
    parser.add_argument("--bits", type=int, default=None, help="Optional override for gptq.bits.")
    parser.add_argument("--group-size", type=int, default=None, help="Optional override for gptq.group_size.")
    parser.add_argument("--calibration-limit", type=int, default=None, help="Optional override for calibration_data.limit.")
    parser.add_argument(
        "--calibration-max-length",
        type=int,
        default=None,
        help="Optional override for calibration_data.max_length and evaluation.perplexity_max_length.",
    )
    args = parser.parse_args()

    config, config_path = load_config(args.config)
    if args.model_path:
        config["model_path"] = args.model_path
    if args.hf_home:
        config["hf_home"] = args.hf_home
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.save_dir:
        config["gptq"]["save_dir"] = args.save_dir
    if args.experiment_name_suffix:
        config["experiment_name"] = f"{config['experiment_name']}_{args.experiment_name_suffix}"
    if args.min_layer is not None:
        config["gptq"]["min_layer"] = args.min_layer
    if args.max_layer is not None:
        config["gptq"]["max_layer"] = args.max_layer
    if args.bits is not None:
        config["gptq"]["bits"] = args.bits
    if args.group_size is not None:
        config["gptq"]["group_size"] = args.group_size
    if args.calibration_limit is not None:
        config["calibration_data"]["limit"] = args.calibration_limit
    if args.calibration_max_length is not None:
        config["calibration_data"]["max_length"] = args.calibration_max_length
        config["evaluation"]["perplexity_max_length"] = args.calibration_max_length

    model_path = resolve_path(REPO_ROOT, config["model_path"])
    hf_home = resolve_path(REPO_ROOT, config.get("hf_home"))
    output_root = resolve_path(REPO_ROOT, config["output_dir"])
    calibration_path = resolve_path(REPO_ROOT, config["calibration_data"].get("path"))
    evaluation_data_cfg = config.get("evaluation_data", {})
    evaluation_path = resolve_path(REPO_ROOT, evaluation_data_cfg.get("path"))
    device_map = config.get("device_map", "").strip()
    quant_mode = config["gptq"]["mode"]

    run_dir = create_run_dir(output_root, "gptq", config["experiment_name"])
    write_json(run_dir / "config_snapshot.json", config)

    if quant_mode == "scaffold":
        tokenizer, model, runtime_device, model_dtype = load_tokenizer_and_model(model_path, device_map, hf_home)
    else:
        tokenizer, model, runtime_device, model_dtype = load_gptq_runtime_model(model_path, hf_home)
        device_map = ""

    inventory = collect_linear_inventory(model)
    targets = select_target_modules(
        inventory,
        config["module_selection"].get("include_groups", []),
        config["module_selection"].get("exclude_patterns", []),
    )
    texts = load_text_samples(calibration_path, config["calibration_data"]["limit"])
    evaluation_texts = load_text_samples(
        evaluation_path if evaluation_data_cfg else calibration_path,
        evaluation_data_cfg.get("limit", config["calibration_data"]["limit"]),
    )

    summary = {
        "method": "gptq",
        "mode": quant_mode,
        "runtime": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "runtime_device": str(runtime_device),
            "torch_dtype": str(model_dtype),
            "device_map": device_map,
        },
        "calibration_samples": len(texts),
        "module_inventory": {
            "total_linear_modules": inventory["total_linear_modules"],
            "total_linear_parameters": inventory["total_linear_parameters"],
            "group_counts": inventory["group_counts"],
            "target_module_count": len(targets),
        },
        "gptq_plan": {
            "bits": config["gptq"]["bits"],
            "group_size": config["gptq"]["group_size"],
            "desc_act": config["gptq"]["desc_act"],
            "damp_percent": config["gptq"]["damp_percent"],
        },
        "source_validation": {
            "upstream_snapshot": str((REPO_ROOT.parent / "external" / "gptq").resolve()),
            "upstream_reference_files": ["gptq.py", "llama.py", "quant.py", "modelutils.py"],
            "scaffold_reuse": [
                "experiments.common.config",
                "experiments.common.data",
                "experiments.common.inventory",
                "experiments.common.reporting",
            ],
            "scaffold_limitations_before_port": [
                "no Hessian accumulation",
                "no forward-hook data capture",
                "no quantizer parameter search",
                "no real in-place weight quantization",
            ],
        },
    }

    if quant_mode == "scaffold":
        if config["evaluation"]["run_generation"]:
            summary["baseline_generation"] = measure_generation(
                model,
                tokenizer,
                texts[: config["evaluation"]["generation_samples"]],
                config["system_prompt"],
                config["evaluation"]["max_new_tokens"],
                runtime_device,
                device_map,
            )

        if config["evaluation"]["run_perplexity"]:
            summary["baseline_perplexity"] = measure_perplexity(
                model,
                tokenizer,
                texts,
                config["evaluation"]["perplexity_max_length"],
                runtime_device,
                device_map,
            )
    else:
        if runtime_device.type != "cuda":
            raise RuntimeError("GPTQ sequential quantization currently requires a CUDA runtime device.")

        calibration_batch = build_calibration_batch(
            tokenizer,
            texts,
            config["system_prompt"],
            int(config["calibration_data"].get("max_length", config["evaluation"]["perplexity_max_length"])),
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

        quant_stats = quantize_openpangu_sequential(
            model,
            calibration_batch["input_ids"],
            calibration_batch["attention_mask"],
            runtime_device,
            {module["name"] for module in targets},
            config["gptq"],
        )
        summary["quant_stats"] = {
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

        write_json(run_dir / "quant_stats.json", quant_stats)
        save_dir = resolve_path(run_dir, config["gptq"].get("save_dir"))
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            summary["saved_model_dir"] = str(save_dir)

    write_json(run_dir / "linear_inventory.json", inventory)
    write_json(run_dir / "target_modules.json", targets)
    write_json(run_dir / "summary.json", summary)

    print(f"[OK] run_dir={run_dir}")
    print(f"[OK] target_module_count={len(targets)}")
    if "baseline_perplexity" in summary:
        print(f"[OK] baseline_perplexity={summary['baseline_perplexity']}")
    if "baseline_generation" in summary:
        print(f"[OK] baseline_generation_tokens_per_second={summary['baseline_generation']['tokens_per_second']}")
    if "quantized_perplexity" in summary:
        print(f"[OK] quantized_perplexity={summary['quantized_perplexity']}")
    if "quant_stats" in summary:
        print(f"[OK] quantized_fraction={summary['quant_stats']['quantized_fraction']}")
    if quant_mode == "scaffold":
        print("[NOTE] GPTQ algorithm application is not implemented in scaffold mode; this stage provides config, inventory, baseline, and reporting scaffolding.")
    else:
        print("[OK] GPTQ sequential quantization port executed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
