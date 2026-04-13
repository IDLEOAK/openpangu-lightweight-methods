import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional


REPO_ROOT = Path(__file__).resolve().parent.parent

FORMAL_PERPLEXITY_KEYS = ["pruned_perplexity", "quantized_perplexity", "baseline_perplexity"]
FORMAL_GENERATION_KEYS = ["pruned_generation", "quantized_generation", "baseline_generation"]
RELOAD_PERPLEXITY_KEYS = ["reloaded_perplexity", "baseline_perplexity"]


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def find_latest_stage_summary(stage_root: Path, method_dir: str) -> Optional[Path]:
    method_root = stage_root / method_dir
    if not method_root.exists():
        return None
    candidates = sorted(
        method_root.glob("*/summary.json"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return None
    return candidates[0]


def read_optional_json(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None or not path.exists():
        return None
    return read_json(path)


def pick_metric(summary: Optional[Dict[str, Any]], keys: list[str]) -> Any:
    if not isinstance(summary, dict):
        return None
    for key in keys:
        if key in summary:
            return summary[key]
    return None


def extract_perplexity_value(metric: Any) -> Optional[float]:
    if isinstance(metric, dict):
        value = metric.get("perplexity")
        if value is None:
            return None
        return float(value)
    if metric is None:
        return None
    return float(metric)


def extract_generation_tps(metric: Any) -> Optional[float]:
    if not isinstance(metric, dict) or metric.get("skipped"):
        return None
    value = metric.get("tokens_per_second")
    if value is None:
        return None
    return float(value)


def extract_size_bytes(export_summary: Optional[Dict[str, Any]], compressed_summary: Optional[Dict[str, Any]]) -> Optional[int]:
    if isinstance(export_summary, dict):
        compressed_info = export_summary.get("compressed_artifact_info", {})
        if isinstance(compressed_info, dict):
            total_size_bytes = compressed_info.get("total_size_bytes")
            if total_size_bytes is not None:
                return int(total_size_bytes)
    if isinstance(compressed_summary, dict):
        artifact_summary = compressed_summary.get("artifact_summary", {})
        if isinstance(artifact_summary, dict):
            total_size_bytes = artifact_summary.get("total_size_bytes")
            if total_size_bytes is not None:
                return int(total_size_bytes)
    return None


def extract_macro_average(summary: Optional[Dict[str, Any]], method_label: str, bucket: str) -> Optional[float]:
    if not isinstance(summary, dict):
        return None
    aggregates = summary.get("aggregates", {})
    macro_average = aggregates.get("macro_average", {})
    method_row = macro_average.get(method_label, {})
    value = method_row.get(bucket)
    if value is None:
        return None
    return float(value)


def extract_reload_delta(summary: Optional[Dict[str, Any]]) -> Optional[float]:
    if not isinstance(summary, dict):
        return None
    source_comparison = summary.get("source_comparison", {})
    value = source_comparison.get("perplexity_delta")
    if value is None:
        return None
    return float(value)


def format_number(value: Optional[float], digits: int = 6) -> str:
    if value is None:
        return "--"
    return f"{value:.{digits}f}"


def format_size_gib(size_bytes: Optional[int]) -> str:
    if size_bytes is None:
        return "--"
    return f"{size_bytes / (1024 ** 3):.2f}"


def collect_variant_row(entry: Dict[str, Any], manifest: Dict[str, Any], results_root: Path) -> Dict[str, Any]:
    formal_config_path = resolve_repo_path(entry["formal_config"])
    wikitext2_config_path = resolve_repo_path(entry["wikitext2_config"])
    cnpublic_config_path = resolve_repo_path(entry["cnpublic_config"])
    formal_config = read_json(formal_config_path)

    variant_id = entry["variant_id"]
    method = entry["method"]
    artifact_method_label = entry.get("artifact_method_label", variant_id)
    variant_root = results_root / entry["result_subdir"]

    export_summary_path = find_latest_stage_summary(variant_root / "export_bundle", method)
    wikitext2_summary_path = find_latest_stage_summary(variant_root / "wikitext2_eval", method)
    cnpublic_summary_path = find_latest_stage_summary(variant_root / "cnpublic_eval", method)
    reload_summary_path = find_latest_stage_summary(variant_root / "reload_verify", "reload_verify")
    compressed_summary_path = find_latest_stage_summary(variant_root / "compressed_verify", "compressed_verify")
    artifact_benchmark_summary_path = variant_root / "artifact_benchmark" / "summary" / "artifact_benchmark_summary.json"

    export_summary = read_optional_json(export_summary_path)
    wikitext2_summary = read_optional_json(wikitext2_summary_path)
    cnpublic_summary = read_optional_json(cnpublic_summary_path)
    reload_summary = read_optional_json(reload_summary_path)
    compressed_summary = read_optional_json(compressed_summary_path)
    artifact_benchmark_summary = read_optional_json(artifact_benchmark_summary_path)

    formal_perplexity = extract_perplexity_value(pick_metric(export_summary, FORMAL_PERPLEXITY_KEYS))
    generation_tps = extract_generation_tps(pick_metric(export_summary, FORMAL_GENERATION_KEYS))
    wikitext2_perplexity = extract_perplexity_value(pick_metric(wikitext2_summary, FORMAL_PERPLEXITY_KEYS))
    cnpublic_perplexity = extract_perplexity_value(pick_metric(cnpublic_summary, FORMAL_PERPLEXITY_KEYS))
    artifact_size_bytes = extract_size_bytes(export_summary, compressed_summary)
    hard8x8_macro_all = extract_macro_average(artifact_benchmark_summary, artifact_method_label, "all")
    hard8x8_macro_en = extract_macro_average(artifact_benchmark_summary, artifact_method_label, "en")
    hard8x8_macro_zh = extract_macro_average(artifact_benchmark_summary, artifact_method_label, "zh")
    reload_b1_delta = extract_reload_delta(reload_summary)
    compressed_b2_delta = extract_reload_delta(compressed_summary)

    missing_stages = []
    if export_summary_path is None:
        missing_stages.append("export_bundle")
    if wikitext2_summary_path is None:
        missing_stages.append("wikitext2_eval")
    if cnpublic_summary_path is None:
        missing_stages.append("cnpublic_eval")
    if reload_summary_path is None:
        missing_stages.append("reload_verify")
    if compressed_summary_path is None:
        missing_stages.append("compressed_verify")
    if not artifact_benchmark_summary_path.exists():
        missing_stages.append("artifact_benchmark")

    module_selection = dict(formal_config.get("module_selection", {}))
    module_selection.update(entry.get("module_selection_overrides", {}))

    calibration_cfg = dict(formal_config.get("calibration_data", {}))
    calibration_cfg.update(entry.get("calibration_overrides", {}))

    method_plan = dict(formal_config.get(method, {}))
    method_plan.update(entry.get("method_overrides", {}))

    row = {
        "variant_id": variant_id,
        "label": entry.get("label", variant_id),
        "method": method,
        "category": manifest.get("category"),
        "axis": manifest.get("axis"),
        "axis_value": entry.get("axis_value"),
        "artifact_method_label": artifact_method_label,
        "result_root": str(variant_root),
        "config_paths": {
            "formal_config": str(formal_config_path),
            "wikitext2_config": str(wikitext2_config_path),
            "cnpublic_config": str(cnpublic_config_path),
        },
        "module_selection": {
            "include_groups": module_selection.get("include_groups", []),
            "exclude_patterns": module_selection.get("exclude_patterns", []),
        },
        "calibration": calibration_cfg,
        "method_config": {
            key: value
            for key, value in method_plan.items()
            if key not in {"mode", "save_dir", "compressed_artifact_dir"}
        },
        "stage_paths": {
            "export_bundle_summary": str(export_summary_path) if export_summary_path else None,
            "wikitext2_summary": str(wikitext2_summary_path) if wikitext2_summary_path else None,
            "cnpublic_summary": str(cnpublic_summary_path) if cnpublic_summary_path else None,
            "reload_verify_summary": str(reload_summary_path) if reload_summary_path else None,
            "compressed_verify_summary": str(compressed_summary_path) if compressed_summary_path else None,
            "artifact_benchmark_summary": str(artifact_benchmark_summary_path) if artifact_benchmark_summary_path.exists() else None,
        },
        "metrics": {
            "formal_eval_perplexity": formal_perplexity,
            "wikitext2_perplexity": wikitext2_perplexity,
            "chinesewebtext2_perplexity": cnpublic_perplexity,
            "hard8x8_macro_average": hard8x8_macro_all,
            "hard8x8_macro_en": hard8x8_macro_en,
            "hard8x8_macro_zh": hard8x8_macro_zh,
            "generation_tokens_per_second": generation_tps,
            "artifact_size_bytes": artifact_size_bytes,
            "reload_b1_perplexity_delta": reload_b1_delta,
            "compressed_b2_perplexity_delta": compressed_b2_delta,
        },
        "missing_stages": missing_stages,
    }
    return row


def build_markdown(payload: Dict[str, Any]) -> str:
    lines = [f"# {payload['study_name']} Summary", ""]
    lines.append(f"- category: `{payload.get('category')}`")
    lines.append(f"- axis: `{payload.get('axis')}`")
    lines.append(f"- results_root: `{payload.get('results_root')}`")
    lines.append("")
    lines.append(
        "| variant | method | axis_value | formal_eval_ppl | "
        "wikitext2_ppl | chinesewebtext2_ppl | hard8x8_macro | "
        "tok/s | artifact_size_gib | b1_delta | b2_delta | missing |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for row in payload["variants"]:
        metrics = row["metrics"]
        missing = ", ".join(row["missing_stages"]) if row["missing_stages"] else "ready"
        lines.append(
            f"| {row['variant_id']} | {row['method']} | {row.get('axis_value')} | "
            f"{format_number(metrics['formal_eval_perplexity'])} | "
            f"{format_number(metrics['wikitext2_perplexity'])} | "
            f"{format_number(metrics['chinesewebtext2_perplexity'])} | "
            f"{format_number(metrics['hard8x8_macro_average'])} | "
            f"{format_number(metrics['generation_tokens_per_second'], digits=4)} | "
            f"{format_size_gib(metrics['artifact_size_bytes'])} | "
            f"{format_number(metrics['reload_b1_perplexity_delta'])} | "
            f"{format_number(metrics['compressed_b2_perplexity_delta'])} | "
            f"{missing} |"
        )
    lines.append("")
    lines.append("## Source Paths")
    lines.append("")
    for row in payload["variants"]:
        lines.append(f"### {row['variant_id']}")
        lines.append("")
        for key, value in row["stage_paths"].items():
            lines.append(f"- {key}: `{value or '--'}`")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate ablation runs across export, public-eval, artifact benchmark, and minimal reload verification.")
    parser.add_argument("--manifest", required=True, help="Path to the ablation variant manifest.")
    parser.add_argument(
        "--results-root",
        default="",
        help="Override for the ablation results root. Defaults to manifest.results_root_suggestion.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directory where the aggregate ablation summary will be written. Defaults to the manifest directory.",
    )
    args = parser.parse_args()

    manifest_path = resolve_repo_path(args.manifest)
    manifest = read_json(manifest_path)
    results_root_value = args.results_root or manifest.get("results_root_suggestion", "")
    if not results_root_value:
        raise ValueError("results root is required either via --results-root or manifest.results_root_suggestion")
    results_root = Path(results_root_value).resolve()

    output_dir = Path(args.output_dir).resolve() if args.output_dir else manifest_path.parent.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "study_name": manifest.get("study_name", manifest_path.stem),
        "category": manifest.get("category"),
        "axis": manifest.get("axis"),
        "results_root": str(results_root),
        "manifest_path": str(manifest_path),
        "variants": [],
    }
    for entry in manifest["variants"]:
        payload["variants"].append(collect_variant_row(entry, manifest, results_root))

    summary_stem = f"{payload['study_name']}_summary"
    summary_json = output_dir / f"{summary_stem}.json"
    summary_md = output_dir / f"{summary_stem}.md"
    write_json(summary_json, payload)
    summary_md.write_text(build_markdown(payload), encoding="utf-8")

    print(f"[OK] summary_json={summary_json}")
    print(f"[OK] summary_md={summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
