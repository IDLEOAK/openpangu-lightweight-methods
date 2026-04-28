import json
from pathlib import Path
from typing import Any, Dict, Optional


EXPERIMENTS_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = EXPERIMENTS_ROOT / "results"


METHOD_SPECS = {
    "sparsegpt": {
        "suffix": "sparsegpt_port_full34_formal_local",
        "plan_key": "sparsegpt_plan",
        "stats_key": "prune_stats",
        "metric_key": "pruned_perplexity",
        "generation_key": "pruned_generation",
        "generation_label": "generation",
        "kind": "prune",
    },
    "admm": {
        "suffix": "admm_port_full34_formal_local",
        "plan_key": "admm_plan",
        "stats_key": "prune_stats",
        "metric_key": "pruned_perplexity",
        "generation_key": "pruned_generation",
        "generation_label": "generation",
        "kind": "prune",
    },
    "llm_bip": {
        "suffix": "llm_bip_port_full34_formal_local",
        "plan_key": "llm_bip_plan",
        "stats_key": "prune_stats",
        "metric_key": "pruned_perplexity",
        "generation_key": None,
        "generation_label": None,
        "kind": "prune",
    },
    "gptq": {
        "suffix": "gptq_port_full34_generation",
        "plan_key": "gptq_plan",
        "stats_key": "quant_stats",
        "metric_key": "quantized_perplexity",
        "generation_key": "quantized_generation",
        "generation_label": "generation",
        "kind": "quant",
    },
    "awq": {
        "suffix": "awq_port_full34_generation",
        "plan_key": "awq_plan",
        "stats_key": "awq_stats",
        "metric_key": "quantized_perplexity",
        "generation_key": "quantized_generation",
        "generation_label": "generation",
        "kind": "quant",
    },
    "smoothquant": {
        "suffix": "smoothquant_port_full34_generation",
        "plan_key": "smoothquant_plan",
        "stats_key": "smoothquant_stats",
        "metric_key": "quantized_perplexity",
        "generation_key": "quantized_generation",
        "generation_label": "generation",
        "kind": "quant",
    },
}


def latest_run(method: str, suffix: str) -> Path:
    base = RESULTS_ROOT / method
    candidates = [p for p in base.iterdir() if p.is_dir() and p.name.endswith(suffix)]
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No run found for {method} with suffix {suffix}")
    return candidates[0]


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pick_first(source: Dict[str, Any], keys: list[str]) -> Optional[Any]:
    for key in keys:
        if key in source:
            return source[key]
    return None


def _format_generation_text(generation: Optional[Dict[str, Any]]) -> str:
    if not generation:
        return "--"
    if generation.get("skipped"):
        return f"skipped ({generation.get('reason', 'unknown')})"
    value = generation.get("tokens_per_second")
    return f"{value} tokens/s" if value is not None else "--"


def _build_method_payload(method: str, run_dir: Path, summary: Dict[str, Any], spec: Dict[str, Any]) -> Dict[str, Any]:
    stats = summary[spec["stats_key"]]
    metric = summary[spec["metric_key"]]
    payload: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "perplexity": metric.get("perplexity"),
        "elapsed_s": stats.get("elapsed_s"),
        "peak_memory_mb": stats.get("peak_memory_mb"),
        "layer_count": stats.get("layer_count"),
        "module_count": stats.get("module_count"),
    }

    plan = summary.get(spec["plan_key"], {})
    if spec["kind"] == "prune":
        payload["total_target_params"] = stats.get("total_pruned_params")
        payload["zero_params"] = stats.get("total_zero_params")
        payload["zero_fraction"] = stats.get("overall_zero_fraction")
        for key in ["sparsity", "pattern", "group_size", "scoring_samples"]:
            if key in plan:
                payload[key] = plan[key]
    else:
        payload["total_target_params"] = stats.get("total_target_params")
        payload["quantized_fraction"] = stats.get("quantized_fraction")
        for key in ["bits", "group_size", "alpha", "alpha_grid", "desc_act", "sym"]:
            if key in plan:
                payload[key] = plan[key]

    generation_key = spec.get("generation_key")
    if generation_key and generation_key in summary:
        payload[spec["generation_label"]] = summary[generation_key]
    return payload


def main() -> int:
    run_dirs = {method: latest_run(method, spec["suffix"]) for method, spec in METHOD_SPECS.items()}
    summaries = {method: read_json(run_dir / "summary.json") for method, run_dir in run_dirs.items()}

    baseline_source = summaries["sparsegpt"]
    baseline_metric = baseline_source["baseline_perplexity"]
    payload: Dict[str, Any] = {
        "baseline": {
            "perplexity": baseline_metric["perplexity"],
            "evaluation_samples": baseline_metric["sample_count"],
            "evaluation_tokens": baseline_metric["token_count"],
            "peak_memory_mb": baseline_metric["peak_memory_mb"],
        }
    }

    for method, spec in METHOD_SPECS.items():
        payload[method] = _build_method_payload(method, run_dirs[method], summaries[method], spec)

    json_path = RESULTS_ROOT / "current_result_summary.json"
    md_path = RESULTS_ROOT / "current_result_summary.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Current Result Summary",
        "",
        "## Scope",
        "",
        "- This snapshot summarizes the current prompt-eval/generation full34 runs across the main methods.",
        "- Public-corpus PPL and benchmark aggregates are maintained in their dedicated summary files.",
        "",
        "## Baseline",
        "",
        f"- evaluation_samples: {payload['baseline']['evaluation_samples']}",
        f"- evaluation_tokens: {payload['baseline']['evaluation_tokens']}",
        f"- perplexity: {payload['baseline']['perplexity']}",
        f"- peak_memory_mb: {payload['baseline']['peak_memory_mb']}",
        "",
        "## Method Snapshot",
        "",
        "| method | route | key setting | PPL | elapsed_s | peak_memory_mb | generation |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]

    route_labels = {
        "sparsegpt": "prune",
        "admm": "prune",
        "llm_bip": "prune-proxy",
        "gptq": "quant",
        "awq": "quant",
        "smoothquant": "quant",
    }
    setting_labels = {
        "sparsegpt": lambda row: f"s={row.get('sparsity')}",
        "admm": lambda row: f"s={row.get('sparsity')}",
        "llm_bip": lambda row: f"s={row.get('sparsity')}, group={row.get('group_size')}",
        "gptq": lambda row: f"{row.get('bits')} bit, g{row.get('group_size')}",
        "awq": lambda row: f"{row.get('bits')} bit, g{row.get('group_size')}",
        "smoothquant": lambda row: f"{row.get('bits')} bit, g{row.get('group_size')}, alpha={row.get('alpha')}",
    }

    for method in ["sparsegpt", "admm", "llm_bip", "gptq", "awq", "smoothquant"]:
        row = payload[method]
        generation = row.get("generation")
        lines.append(
            f"| {method} | {route_labels[method]} | {setting_labels[method](row)} | "
            f"{row.get('perplexity')} | {row.get('elapsed_s')} | {row.get('peak_memory_mb')} | {_format_generation_text(generation)} |"
        )

    lines.extend(
        [
            "",
            "## Run Directories",
            "",
        ]
    )
    for method in ["sparsegpt", "admm", "llm_bip", "gptq", "awq", "smoothquant"]:
        lines.append(f"- {method}: `{payload[method]['run_dir']}`")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] json={json_path}")
    print(f"[OK] markdown={md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
