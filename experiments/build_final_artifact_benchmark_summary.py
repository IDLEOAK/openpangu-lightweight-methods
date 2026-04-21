import json
from pathlib import Path


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def collect_latest_runs(results_roots: list[Path], methods: list[str]):
    latest = {}
    for results_root in results_roots:
        for method in methods:
            method_dir = results_root / method
            if not method_dir.exists():
                continue
            for run_dir in method_dir.iterdir():
                summary_path = run_dir / "summary.json"
                if not summary_path.exists():
                    continue
                summary = read_json(summary_path)
                benchmark_plan = summary.get("benchmark_plan", {})
                task_slug = benchmark_plan.get("task_slug")
                if not task_slug:
                    continue
                key = (method, task_slug)
                mtime = summary_path.stat().st_mtime
                current = latest.get(key)
                if current is None or mtime > current["mtime"]:
                    benchmark_result = summary.get("benchmark_result", {})
                    latest[key] = {
                        "method": method,
                        "task_slug": task_slug,
                        "run_dir": str(run_dir),
                        "accuracy": benchmark_result.get("accuracy"),
                        "evaluated_count": benchmark_result.get("evaluated_count"),
                        "correct_count": benchmark_result.get("correct_count"),
                        "mtime": mtime,
                    }
    return latest


def build_aggregates(tasks_payload: dict, task_language: dict, methods: list[str]):
    aggregates = {
        "macro_average": {},
        "weighted_average": {},
    }
    for method in methods:
        macro = {"all": [], "en": [], "zh": []}
        weighted = {
            "all": {"correct": 0, "evaluated": 0},
            "en": {"correct": 0, "evaluated": 0},
            "zh": {"correct": 0, "evaluated": 0},
        }
        for task_slug, method_rows in tasks_payload.items():
            row = method_rows.get(method)
            if not row:
                continue
            lang = task_language.get(task_slug, "unknown")
            macro["all"].append(row["accuracy"])
            weighted["all"]["correct"] += row["correct_count"]
            weighted["all"]["evaluated"] += row["evaluated_count"]
            if lang in ("en", "zh"):
                macro[lang].append(row["accuracy"])
                weighted[lang]["correct"] += row["correct_count"]
                weighted[lang]["evaluated"] += row["evaluated_count"]
        aggregates["macro_average"][method] = {
            bucket: (round(sum(vals) / len(vals), 6) if vals else None)
            for bucket, vals in macro.items()
        }
        aggregates["weighted_average"][method] = {
            bucket: (
                round(weighted[bucket]["correct"] / weighted[bucket]["evaluated"], 6)
                if weighted[bucket]["evaluated"]
                else None
            )
            for bucket in ("all", "en", "zh")
        }
    return aggregates


def resolve_from_repo_root(repo_root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    else:
        path = path.resolve()
    return path


def discover_results_roots(results_root: Path) -> list[Path]:
    roots: list[Path] = []
    if results_root.exists():
        roots.append(results_root)

    # Local synced snapshots are typically returned as final_artifact_hard_8x8_<timestamp>/.
    for snapshot_root in sorted(
        (path for path in results_root.parent.glob("final_artifact_hard_8x8_*") if path.is_dir()),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    ):
        if snapshot_root not in roots:
            roots.append(snapshot_root)
    return roots


def main() -> int:
    experiments_root = Path(__file__).resolve().parent
    repo_root = experiments_root.parent
    task_manifest = read_json(experiments_root / "configs" / "final_artifact_benchmark_tasks.json")
    method_manifest = read_json(experiments_root / "configs" / "final_artifact_benchmark_models.json")
    results_root = resolve_from_repo_root(repo_root, method_manifest["final_results_root"])
    discovered_results_roots = discover_results_roots(results_root)
    summary_root = resolve_from_repo_root(repo_root, method_manifest["final_summary_root"])
    repo_summary_root = results_root.parent
    output_roots: list[Path] = []
    for root in (summary_root, repo_summary_root):
        if root not in output_roots:
            root.mkdir(parents=True, exist_ok=True)
            output_roots.append(root)

    methods = ["baseline"] + list(method_manifest["methods"].keys())
    task_entries = task_manifest["tasks"]
    task_language = {Path(item["benchmark_data"]).stem: item["language"] for item in task_entries}
    ordered_tasks = [Path(item["benchmark_data"]).stem for item in task_entries]
    task_counts = {
        "all": len(ordered_tasks),
        "en": sum(1 for item in task_entries if item["language"] == "en"),
        "zh": sum(1 for item in task_entries if item["language"] == "zh"),
    }

    latest = collect_latest_runs(discovered_results_roots, methods)
    if not latest:
        searched = ", ".join(str(path) for path in discovered_results_roots) or str(results_root)
        raise RuntimeError(f"No final-artifact benchmark runs found under: {searched}")
    payload = {"tasks": {}, "methods": methods}
    for task_slug in ordered_tasks:
        payload["tasks"][task_slug] = {}
        for method in methods:
            row = latest.get((method, task_slug))
            if row:
                payload["tasks"][task_slug][method] = {
                    "accuracy": row["accuracy"],
                    "evaluated_count": row["evaluated_count"],
                    "correct_count": row["correct_count"],
                    "run_dir": row["run_dir"],
                }

    payload["aggregates"] = build_aggregates(payload["tasks"], task_language, methods)

    json_text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    summary_json_paths = [root / "final_artifact_benchmark_summary.json" for root in output_roots]
    summary_md_paths = [root / "final_artifact_benchmark_summary.md" for root in output_roots]
    for path in summary_json_paths:
        path.write_text(json_text, encoding="utf-8")

    lines = ["# Final Artifact Benchmark Summary", ""]
    lines.append("## Aggregate Metrics")
    lines.append("")
    lines.append(
        f"| method | macro(all {task_counts['all']}) | weighted(all) | "
        f"macro(en {task_counts['en']}) | weighted(en) | "
        f"macro(zh {task_counts['zh']}) | weighted(zh) |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for method in methods:
        macro = payload["aggregates"]["macro_average"][method]
        weighted = payload["aggregates"]["weighted_average"][method]
        lines.append(
            f"| {method} | {macro['all']} | {weighted['all']} | {macro['en']} | {weighted['en']} | {macro['zh']} | {weighted['zh']} |"
        )
    lines.append("")
    for task_slug in ordered_tasks:
        lines.append(f"## {task_slug}")
        lines.append("")
        lines.append("| method | accuracy | correct/evaluated | run_dir |")
        lines.append("| --- | ---: | ---: | --- |")
        for method in methods:
            row = payload["tasks"][task_slug].get(method)
            if not row:
                lines.append(f"| {method} | -- | -- | -- |")
                continue
            lines.append(
                f"| {method} | {row['accuracy']} | {row['correct_count']}/{row['evaluated_count']} | `{row['run_dir']}` |"
            )
        lines.append("")
    markdown_text = "\n".join(lines) + "\n"
    for path in summary_md_paths:
        path.write_text(markdown_text, encoding="utf-8")
    for path in summary_json_paths:
        print(f"[OK] json={path}")
    for path in summary_md_paths:
        print(f"[OK] markdown={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
