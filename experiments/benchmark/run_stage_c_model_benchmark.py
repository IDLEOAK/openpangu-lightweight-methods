import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Dict

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.common.benchmark import evaluate_multiple_choice, load_multiple_choice_samples
from experiments.common.config import load_config, resolve_path
from experiments.common.reporting import create_run_dir, summarize_directory, write_json
from experiments.common.runtime import load_tokenizer_and_model
from experiments.compressed_artifacts.io import load_quant_artifact, load_sparse_artifact


def read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one Stage C benchmark task for either the dense baseline or a Stage B compressed artifact.")
    parser.add_argument("--config", required=True, help="Path to Stage C benchmark config.")
    parser.add_argument("--benchmark-data", default="", help="Optional override for config.benchmark_data.path.")
    parser.add_argument("--limit", type=int, default=None, help="Optional override for config.benchmark_data.limit.")
    parser.add_argument("--max-length", type=int, default=None, help="Optional override for config.evaluation.max_length.")
    parser.add_argument("--output-dir", default="", help="Optional override for config.output_dir.")
    parser.add_argument("--hf-home", default="", help="Optional override for config.hf_home.")
    parser.add_argument("--method-label", default="", help="Optional override for config.method_label.")
    parser.add_argument("--artifact-dir", default="", help="Optional compressed artifact directory for non-baseline runs.")
    parser.add_argument("--experiment-name-suffix", default="", help="Optional suffix appended to experiment_name.")
    args = parser.parse_args()

    config, _ = load_config(args.config)
    if args.benchmark_data:
        config["benchmark_data"]["path"] = args.benchmark_data
    if args.limit is not None:
        config["benchmark_data"]["limit"] = args.limit
    if args.max_length is not None:
        config["evaluation"]["max_length"] = args.max_length
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.hf_home:
        config["hf_home"] = args.hf_home
    if args.method_label:
        config["method_label"] = args.method_label
    if args.artifact_dir:
        config["artifact_dir"] = args.artifact_dir
    if args.experiment_name_suffix:
        config["experiment_name"] = f"{config['experiment_name']}_{args.experiment_name_suffix}"

    model_path = resolve_path(REPO_ROOT, config["model_path"])
    hf_home = resolve_path(REPO_ROOT, config.get("hf_home"))
    output_root = resolve_path(REPO_ROOT, config["output_dir"])
    benchmark_path = resolve_path(REPO_ROOT, config["benchmark_data"]["path"])
    method_label = config.get("method_label", "baseline").strip() or "baseline"
    artifact_dir = resolve_path(REPO_ROOT, config.get("artifact_dir"))

    samples = load_multiple_choice_samples(benchmark_path, int(config["benchmark_data"].get("limit", 0)))
    task_slug = benchmark_path.stem
    run_dir = create_run_dir(output_root, method_label, f"{config['experiment_name']}_{task_slug}")
    write_json(run_dir / "config_snapshot.json", config)

    tokenizer, model, runtime_device, model_dtype = load_tokenizer_and_model(model_path, "", hf_home)
    artifact_manifest = None
    if artifact_dir is not None:
        artifact_manifest = read_json(artifact_dir / "manifest.json")
        if artifact_manifest["artifact_type"] == "sparse_overlay":
            load_sparse_artifact(model, artifact_dir)
        elif artifact_manifest["artifact_type"] == "packed_quant_overlay":
            load_quant_artifact(model, artifact_dir)
        else:
            raise RuntimeError(f"Unsupported artifact_type={artifact_manifest['artifact_type']}")
        model.to(runtime_device)
        model.eval()

    summary = {
        "method": method_label,
        "benchmark_data_path": str(benchmark_path),
        "runtime": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "runtime_device": str(runtime_device),
            "torch_dtype": str(model_dtype),
            "device_map": "",
        },
        "benchmark_plan": {
            "task_slug": task_slug,
            "limit": len(samples),
            "max_length": int(config["evaluation"]["max_length"]),
            "scoring_mode": config["evaluation"].get("scoring_mode", "avg_logprob"),
        },
    }
    if artifact_dir is not None:
        summary["artifact_dir"] = str(artifact_dir)
        summary["artifact_manifest"] = artifact_manifest
        summary["artifact_summary"] = summarize_directory(artifact_dir)

    benchmark_result = evaluate_multiple_choice(
        model,
        tokenizer,
        samples,
        config["system_prompt"],
        runtime_device,
        "",
        int(config["evaluation"]["max_length"]),
        config["evaluation"].get("scoring_mode", "avg_logprob"),
    )
    summary["benchmark_result"] = {
        "sample_count": benchmark_result["sample_count"],
        "evaluated_count": benchmark_result["evaluated_count"],
        "skipped_count": benchmark_result["skipped_count"],
        "correct_count": benchmark_result["correct_count"],
        "accuracy": benchmark_result["accuracy"],
        "scoring_mode": benchmark_result["scoring_mode"],
        "tasks": benchmark_result["tasks"],
    }

    write_json(run_dir / "benchmark_predictions.json", benchmark_result["results"])
    write_json(run_dir / "summary.json", summary)

    print(f"[OK] run_dir={run_dir}")
    print(f"[OK] method={method_label}")
    print(f"[OK] benchmark_data={benchmark_path}")
    print(f"[OK] accuracy={benchmark_result['accuracy']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
