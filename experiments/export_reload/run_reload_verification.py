import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.common.benchmark import evaluate_multiple_choice, load_multiple_choice_samples
from experiments.common.config import load_config, resolve_path
from experiments.common.data import load_text_samples
from experiments.common.openpangu_sequential import build_calibration_batch, evaluate_openpangu_perplexity_sequential
from experiments.common.reporting import create_run_dir, summarize_directory, write_json
from experiments.common.runtime import load_tokenizer_and_model


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pick_source_metric(source_summary: Dict[str, Any], keys: list[str]) -> Optional[Any]:
    for key in keys:
        if key in source_summary:
            return source_summary[key]
    return None


def _extract_perplexity_value(metric: Any) -> Optional[float]:
    if isinstance(metric, dict):
        value = metric.get("perplexity")
        if value is None:
            return None
        return float(value)
    if metric is None:
        return None
    return float(metric)


def main() -> int:
    parser = argparse.ArgumentParser(description="Reload an exported OpenPangu artifact and run minimal PPL + benchmark verification.")
    parser.add_argument("--config", required=True, help="Path to reload verification JSON config.")
    parser.add_argument("--model-path", default="", help="Optional override for config.model_path.")
    parser.add_argument("--hf-home", default="", help="Optional override for config.hf_home.")
    parser.add_argument("--output-dir", default="", help="Optional override for config.output_dir.")
    parser.add_argument("--benchmark-data", default="", help="Optional override for config.benchmark_data.path.")
    parser.add_argument("--benchmark-limit", type=int, default=None, help="Optional override for config.benchmark_data.limit.")
    parser.add_argument("--benchmark-max-length", type=int, default=None, help="Optional override for config.benchmark_evaluation.max_length.")
    parser.add_argument("--evaluation-data", default="", help="Optional override for config.evaluation_data.path.")
    parser.add_argument("--evaluation-limit", type=int, default=None, help="Optional override for config.evaluation_data.limit.")
    parser.add_argument("--evaluation-max-length", type=int, default=None, help="Optional override for config.evaluation.perplexity_max_length.")
    parser.add_argument("--experiment-name-suffix", default="", help="Optional suffix appended to experiment_name.")
    parser.add_argument("--source-summary", default="", help="Optional path to the in-memory run summary.json for comparison.")
    args = parser.parse_args()

    config, _ = load_config(args.config)
    if args.model_path:
        config["model_path"] = args.model_path
    if args.hf_home:
        config["hf_home"] = args.hf_home
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.benchmark_data:
        config["benchmark_data"]["path"] = args.benchmark_data
    if args.benchmark_limit is not None:
        config["benchmark_data"]["limit"] = args.benchmark_limit
    if args.benchmark_max_length is not None:
        config["benchmark_evaluation"]["max_length"] = args.benchmark_max_length
    if args.evaluation_data:
        config["evaluation_data"]["path"] = args.evaluation_data
    if args.evaluation_limit is not None:
        config["evaluation_data"]["limit"] = args.evaluation_limit
    if args.evaluation_max_length is not None:
        config["evaluation"]["perplexity_max_length"] = args.evaluation_max_length
        config["evaluation_data"]["max_length"] = args.evaluation_max_length
    if args.experiment_name_suffix:
        config["experiment_name"] = f"{config['experiment_name']}_{args.experiment_name_suffix}"

    model_path = resolve_path(REPO_ROOT, config["model_path"])
    hf_home = resolve_path(REPO_ROOT, config.get("hf_home"))
    output_root = resolve_path(REPO_ROOT, config["output_dir"])
    evaluation_path = resolve_path(REPO_ROOT, config["evaluation_data"]["path"])
    benchmark_path = resolve_path(REPO_ROOT, config["benchmark_data"]["path"])
    device_map = config.get("device_map", "").strip()

    run_dir = create_run_dir(output_root, "reload_verify", config["experiment_name"])
    write_json(run_dir / "config_snapshot.json", config)

    tokenizer, model, runtime_device, model_dtype = load_tokenizer_and_model(model_path, device_map, hf_home)
    evaluation_texts = load_text_samples(evaluation_path, int(config["evaluation_data"].get("limit", 0)))
    benchmark_samples = load_multiple_choice_samples(benchmark_path, int(config["benchmark_data"].get("limit", 0)))
    benchmark_task_slug = benchmark_path.stem

    evaluation_batch = build_calibration_batch(
        tokenizer,
        evaluation_texts,
        config["system_prompt"],
        int(config["evaluation"]["perplexity_max_length"]),
        apply_chat_template=not bool(config["evaluation_data"].get("raw_text", False)),
    )

    summary: Dict[str, Any] = {
        "method": "reload_verify",
        "reloaded_model_path": str(model_path),
        "artifact_summary": summarize_directory(model_path),
        "runtime": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "runtime_device": str(runtime_device),
            "torch_dtype": str(model_dtype),
            "device_map": device_map,
        },
        "evaluation_plan": {
            "path": str(evaluation_path),
            "sample_count": len(evaluation_texts),
            "perplexity_max_length": int(config["evaluation"]["perplexity_max_length"]),
            "token_count": int(evaluation_batch["attention_mask"].sum().item()) if evaluation_batch.get("attention_mask") is not None else None,
        },
        "benchmark_plan": {
            "path": str(benchmark_path),
            "task_slug": benchmark_task_slug,
            "sample_count": len(benchmark_samples),
            "max_length": int(config["benchmark_evaluation"]["max_length"]),
            "scoring_mode": config["benchmark_evaluation"].get("scoring_mode", "avg_logprob"),
        },
    }

    reloaded_perplexity = evaluate_openpangu_perplexity_sequential(
        model,
        evaluation_batch["input_ids"],
        evaluation_batch["attention_mask"],
        runtime_device,
    )
    summary["reloaded_perplexity"] = reloaded_perplexity

    model.to(runtime_device)
    model.eval()
    benchmark_result = evaluate_multiple_choice(
        model,
        tokenizer,
        benchmark_samples,
        config["system_prompt"],
        runtime_device,
        device_map,
        int(config["benchmark_evaluation"]["max_length"]),
        config["benchmark_evaluation"].get("scoring_mode", "avg_logprob"),
    )
    summary["reloaded_benchmark"] = {
        "sample_count": benchmark_result["sample_count"],
        "evaluated_count": benchmark_result["evaluated_count"],
        "skipped_count": benchmark_result["skipped_count"],
        "correct_count": benchmark_result["correct_count"],
        "accuracy": benchmark_result["accuracy"],
        "scoring_mode": benchmark_result["scoring_mode"],
        "tasks": benchmark_result["tasks"],
    }
    write_json(run_dir / "benchmark_predictions.json", benchmark_result["results"])

    if args.source_summary:
        source_summary_path = Path(args.source_summary).resolve()
        source_summary = _read_json(source_summary_path)
        source_perplexity = _pick_source_metric(
            source_summary,
            ["pruned_perplexity", "quantized_perplexity", "baseline_perplexity"],
        )
        source_benchmark = _pick_source_metric(
            source_summary,
            ["pruned_benchmark", "quantized_benchmark", "baseline_benchmark"],
        )
        comparison: Dict[str, Any] = {
            "source_summary_path": str(source_summary_path),
        }
        if source_perplexity is not None:
            comparison["source_perplexity"] = source_perplexity
            source_perplexity_value = _extract_perplexity_value(source_perplexity)
            reloaded_perplexity_value = _extract_perplexity_value(reloaded_perplexity)
            if source_perplexity_value is not None and reloaded_perplexity_value is not None:
                comparison["perplexity_delta"] = reloaded_perplexity_value - source_perplexity_value
        if isinstance(source_benchmark, dict) and "accuracy" in source_benchmark:
            comparison["source_benchmark_accuracy"] = source_benchmark["accuracy"]
            comparison["benchmark_accuracy_delta"] = benchmark_result["accuracy"] - source_benchmark["accuracy"]
        if "saved_model_format" in source_summary:
            comparison["saved_model_format"] = source_summary["saved_model_format"]
        summary["source_comparison"] = comparison

    write_json(run_dir / "summary.json", summary)

    print(f"[OK] run_dir={run_dir}")
    print(f"[OK] reloaded_model_path={model_path}")
    print(f"[OK] total_size_bytes={summary['artifact_summary']['total_size_bytes']}")
    print(f"[OK] reloaded_perplexity={summary['reloaded_perplexity']}")
    print(f"[OK] reloaded_benchmark_accuracy={summary['reloaded_benchmark']['accuracy']}")
    if "source_comparison" in summary:
        if "perplexity_delta" in summary["source_comparison"]:
            print(f"[OK] perplexity_delta={summary['source_comparison']['perplexity_delta']}")
        if "benchmark_accuracy_delta" in summary["source_comparison"]:
            print(f"[OK] benchmark_accuracy_delta={summary['source_comparison']['benchmark_accuracy_delta']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
