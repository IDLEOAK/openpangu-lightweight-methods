import argparse
import platform
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.common.config import load_config, resolve_path
from experiments.common.data import load_text_samples
from experiments.common.inventory import collect_linear_inventory, select_target_modules
from experiments.common.metrics import measure_generation, measure_perplexity
from experiments.common.reporting import create_run_dir, write_json
from experiments.common.runtime import load_tokenizer_and_model


def main() -> int:
    parser = argparse.ArgumentParser(description="GPTQ scaffold for OpenPangu.")
    parser.add_argument("--config", required=True, help="Path to experiment JSON config.")
    args = parser.parse_args()

    config, config_path = load_config(args.config)
    model_path = resolve_path(REPO_ROOT, config["model_path"])
    hf_home = resolve_path(REPO_ROOT, config.get("hf_home"))
    output_root = resolve_path(REPO_ROOT, config["output_dir"])
    calibration_path = resolve_path(REPO_ROOT, config["calibration_data"].get("path"))
    device_map = config.get("device_map", "").strip()

    run_dir = create_run_dir(output_root, "gptq", config["experiment_name"])
    write_json(run_dir / "config_snapshot.json", config)

    tokenizer, model, runtime_device, model_dtype = load_tokenizer_and_model(model_path, device_map, hf_home)
    inventory = collect_linear_inventory(model)
    targets = select_target_modules(
        inventory,
        config["module_selection"].get("include_groups", []),
        config["module_selection"].get("exclude_patterns", []),
    )
    texts = load_text_samples(calibration_path, config["calibration_data"]["limit"])

    summary = {
        "method": "gptq",
        "mode": config["gptq"]["mode"],
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
    }

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

    write_json(run_dir / "linear_inventory.json", inventory)
    write_json(run_dir / "target_modules.json", targets)
    write_json(run_dir / "summary.json", summary)

    print(f"[OK] run_dir={run_dir}")
    print(f"[OK] target_module_count={len(targets)}")
    if "baseline_perplexity" in summary:
        print(f"[OK] baseline_perplexity={summary['baseline_perplexity']}")
    if "baseline_generation" in summary:
        print(f"[OK] baseline_generation_tokens_per_second={summary['baseline_generation']['tokens_per_second']}")
    print("[NOTE] GPTQ algorithm application is not implemented yet; this stage provides config, inventory, baseline, and reporting scaffolding.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
