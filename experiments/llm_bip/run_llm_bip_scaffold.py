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
from experiments.common.inventory import collect_linear_inventory, select_target_modules, validate_target_modules
from experiments.common.reporting import create_run_dir, write_json
from experiments.common.runtime import ensure_hf_home, select_runtime
from experiments.llm_bip.algorithm import (
    build_calibration_batch,
    evaluate_openpangu_perplexity_sequential,
    prune_openpangu_llm_bip_sequential,
)


def load_runtime_model(model_path: Path, hf_home: Optional[Path]):
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
    parser = argparse.ArgumentParser(description="LLM-BIP structured pruning scaffold for OpenPangu.")
    parser.add_argument("--config", required=True, help="Path to experiment JSON config.")
    parser.add_argument("--model-path", default="", help="Optional override for config.model_path.")
    parser.add_argument("--hf-home", default="", help="Optional override for config.hf_home.")
    parser.add_argument("--output-dir", default="", help="Optional override for config.output_dir.")
    parser.add_argument("--experiment-name-suffix", default="", help="Optional suffix appended to experiment_name.")
    parser.add_argument("--min-layer", type=int, default=None, help="Optional override for llm_bip.min_layer.")
    parser.add_argument("--max-layer", type=int, default=None, help="Optional override for llm_bip.max_layer.")
    parser.add_argument("--sparsity", type=float, default=None, help="Optional override for llm_bip.sparsity.")
    parser.add_argument("--group-size", type=int, default=None, help="Optional override for llm_bip.group_size.")
    parser.add_argument("--scoring-samples", type=int, default=None, help="Optional override for llm_bip.scoring_samples.")
    parser.add_argument("--calibration-limit", type=int, default=None, help="Optional override for calibration_data.limit.")
    parser.add_argument(
        "--calibration-max-length",
        type=int,
        default=None,
        help="Optional override for calibration_data.max_length and evaluation.perplexity_max_length.",
    )
    args = parser.parse_args()

    config, _ = load_config(args.config)
    if args.model_path:
        config["model_path"] = args.model_path
    if args.hf_home:
        config["hf_home"] = args.hf_home
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.experiment_name_suffix:
        config["experiment_name"] = f"{config['experiment_name']}_{args.experiment_name_suffix}"
    if args.min_layer is not None:
        config["llm_bip"]["min_layer"] = args.min_layer
    if args.max_layer is not None:
        config["llm_bip"]["max_layer"] = args.max_layer
    if args.sparsity is not None:
        config["llm_bip"]["sparsity"] = args.sparsity
    if args.group_size is not None:
        config["llm_bip"]["group_size"] = args.group_size
    if args.scoring_samples is not None:
        config["llm_bip"]["scoring_samples"] = args.scoring_samples
    if args.calibration_limit is not None:
        config["calibration_data"]["limit"] = args.calibration_limit
    if args.calibration_max_length is not None:
        config["calibration_data"]["max_length"] = args.calibration_max_length
        config["evaluation"]["perplexity_max_length"] = args.calibration_max_length

    if config["evaluation"].get("run_generation", False):
        raise ValueError(
            "LLM-BIP proxy runner does not implement generation benchmarking. "
            "Set evaluation.run_generation=false."
        )

    model_path = resolve_path(REPO_ROOT, config["model_path"])
    hf_home = resolve_path(REPO_ROOT, config.get("hf_home"))
    output_root = resolve_path(REPO_ROOT, config["output_dir"])
    calibration_path = resolve_path(REPO_ROOT, config["calibration_data"].get("path"))
    evaluation_data_cfg = config.get("evaluation_data", {})
    evaluation_path = resolve_path(REPO_ROOT, evaluation_data_cfg.get("path"))

    run_dir = create_run_dir(output_root, "llm_bip", config["experiment_name"])
    write_json(run_dir / "config_snapshot.json", config)

    tokenizer, model, runtime_device, model_dtype = load_runtime_model(model_path, hf_home)
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

    summary = {
        "method": "llm_bip",
        "mode": config["llm_bip"]["mode"],
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
        "llm_bip_plan": {
            "sparsity": config["llm_bip"]["sparsity"],
            "group_size": config["llm_bip"]["group_size"],
            "scoring_samples": config["llm_bip"]["scoring_samples"],
        },
        "source_validation": {
            "paper_reference": "Wu et al. 2024 LLM-BIP",
            "expected_upstream_snapshot": str((REPO_ROOT.parent / "external" / "llm-bip").resolve()),
            "current_scope_note": "current local port uses block-output reconstruction score proxy with structured output-channel groups",
            "scaffold_reuse": [
                "experiments.common.config",
                "experiments.common.data",
                "experiments.common.inventory",
                "experiments.common.reporting",
            ],
        },
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

    prune_stats = prune_openpangu_llm_bip_sequential(
        model,
        calibration_batch["input_ids"],
        calibration_batch["attention_mask"],
        runtime_device,
        {module["name"] for module in targets},
        config["llm_bip"],
    )
    summary["prune_stats"] = {
        "module_count": prune_stats["module_count"],
        "total_zero_params": prune_stats["total_zero_params"],
        "total_pruned_params": prune_stats["total_pruned_params"],
        "overall_zero_fraction": prune_stats["overall_zero_fraction"],
        "layer_count": len(prune_stats["layers"]),
        "elapsed_s": prune_stats["elapsed_s"],
        "peak_memory_mb": prune_stats["peak_memory_mb"],
    }

    if config["evaluation"].get("run_post_prune_perplexity", True):
        summary["pruned_perplexity"] = evaluate_openpangu_perplexity_sequential(
            model,
            evaluation_batch["input_ids"],
            evaluation_batch["attention_mask"],
            runtime_device,
        )

    write_json(run_dir / "linear_inventory.json", inventory)
    write_json(run_dir / "target_modules.json", targets)
    write_json(run_dir / "prune_stats.json", prune_stats)
    write_json(run_dir / "summary.json", summary)

    print(f"[OK] run_dir={run_dir}")
    print(f"[OK] target_module_count={len(targets)}")
    if "baseline_perplexity" in summary:
        print(f"[OK] baseline_perplexity={summary['baseline_perplexity']}")
    if "pruned_perplexity" in summary:
        print(f"[OK] pruned_perplexity={summary['pruned_perplexity']}")
    print("[OK] LLM-BIP structured pruning proxy port executed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
