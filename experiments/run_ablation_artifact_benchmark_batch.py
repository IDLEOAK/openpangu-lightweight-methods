import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def latest_run(output_root: Path, method_label: str, experiment_name: str, task_slug: str) -> Path:
    method_dir = output_root / method_label
    pattern = f"*{experiment_name}_{task_slug}"
    matches = sorted(
        method_dir.glob(pattern),
        key=lambda item: (item / "summary.json").stat().st_mtime if (item / "summary.json").exists() else item.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise RuntimeError(f"Run dir not found for method_label={method_label} task={task_slug}")
    return matches[0]


def collect_latest_runs(results_root: Path, method_label: str):
    latest = {}
    method_dir = results_root / method_label
    if not method_dir.exists():
        return latest
    for run_dir in method_dir.iterdir():
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue
        summary = read_json(summary_path)
        benchmark_plan = summary.get("benchmark_plan", {})
        task_slug = benchmark_plan.get("task_slug")
        if not task_slug:
            continue
        mtime = summary_path.stat().st_mtime
        current = latest.get(task_slug)
        if current is None or mtime > current["mtime"]:
            benchmark_result = summary.get("benchmark_result", {})
            latest[task_slug] = {
                "method": method_label,
                "task_slug": task_slug,
                "run_dir": str(run_dir),
                "summary_path": str(summary_path),
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
            bucket: (round(sum(values) / len(values), 6) if values else None)
            for bucket, values in macro.items()
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


def build_markdown(summary_payload: dict, ordered_tasks: list[str], task_counts: dict) -> str:
    methods = summary_payload["methods"]
    lines = ["# Artifact Benchmark Summary", ""]
    lines.append(f"- task_group: `{summary_payload.get('task_group')}`")
    lines.append(f"- artifact_dir: `{summary_payload.get('artifact_dir')}`")
    lines.append("")
    lines.append("## Aggregate Metrics")
    lines.append("")
    lines.append(
        f"| method | macro(all {task_counts['all']}) | weighted(all) | "
        f"macro(en {task_counts['en']}) | weighted(en) | "
        f"macro(zh {task_counts['zh']}) | weighted(zh) |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for method in methods:
        macro = summary_payload["aggregates"]["macro_average"][method]
        weighted = summary_payload["aggregates"]["weighted_average"][method]
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
            row = summary_payload["tasks"][task_slug].get(method)
            if not row:
                lines.append(f"| {method} | -- | -- | -- |")
                continue
            lines.append(
                f"| {method} | {row['accuracy']} | {row['correct_count']}/{row['evaluated_count']} | `{row['run_dir']}` |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the hard-8x8 final-artifact benchmark for one ablation variant.")
    parser.add_argument("--method-label", required=True, help="Label used for this ablation variant, e.g. sparsegpt_s020.")
    parser.add_argument("--artifact-dir", required=True, help="Compressed artifact directory for this variant.")
    parser.add_argument("--result-root", required=True, help="Variant-level result root, e.g. /mnt/results/ablation/pruning/sparsegpt_s020.")
    parser.add_argument("--python-bin", default=sys.executable, help="Python interpreter used to launch benchmark subtasks.")
    parser.add_argument(
        "--benchmark-config",
        default="experiments/configs/final_artifact_benchmark_base.json",
        help="Path to the shared final-artifact benchmark base config.",
    )
    parser.add_argument(
        "--task-manifest",
        default="experiments/configs/final_artifact_benchmark_tasks.json",
        help="Path to the hard-8x8 benchmark task manifest.",
    )
    parser.add_argument(
        "--summary-stem",
        default="artifact_benchmark_summary",
        help="Filename stem used for the per-variant aggregate summary.",
    )
    parser.add_argument("--hf-home", default="", help="Optional override passed through to the task runner.")
    args = parser.parse_args()

    benchmark_config = resolve_repo_path(args.benchmark_config)
    task_manifest_path = resolve_repo_path(args.task_manifest)
    artifact_dir = Path(args.artifact_dir).resolve()
    result_root = Path(args.result_root).resolve()

    task_manifest = read_json(task_manifest_path)
    benchmark_config_payload = read_json(benchmark_config)
    experiment_name = benchmark_config_payload["experiment_name"]

    benchmark_root = result_root / "artifact_benchmark"
    process_root = benchmark_root / "process"
    final_results_root = benchmark_root / "final"
    summary_root = benchmark_root / "summary"
    process_root.mkdir(parents=True, exist_ok=True)
    final_results_root.mkdir(parents=True, exist_ok=True)
    summary_root.mkdir(parents=True, exist_ok=True)

    runner = REPO_ROOT / "experiments" / "benchmark" / "run_final_artifact_model_benchmark.py"
    for task in task_manifest["tasks"]:
        benchmark_data = task["benchmark_data"]
        task_slug = Path(benchmark_data).stem
        cmd = [
            args.python_bin,
            str(runner),
            "--config",
            str(benchmark_config),
            "--benchmark-data",
            benchmark_data,
            "--output-dir",
            str(process_root),
            "--method-label",
            args.method_label,
            "--artifact-dir",
            str(artifact_dir),
        ]
        if args.hf_home:
            cmd.extend(["--hf-home", args.hf_home])
        print("[RUN]", args.method_label, task_slug)
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)
        src_run = latest_run(process_root, args.method_label, experiment_name, task_slug)
        dst_method_root = final_results_root / args.method_label
        dst_method_root.mkdir(parents=True, exist_ok=True)
        dst_run = dst_method_root / src_run.name
        if dst_run.exists():
            shutil.rmtree(dst_run)
        shutil.copytree(src_run, dst_run)

    ordered_tasks = [Path(item["benchmark_data"]).stem for item in task_manifest["tasks"]]
    task_language = {Path(item["benchmark_data"]).stem: item["language"] for item in task_manifest["tasks"]}
    task_counts = {
        "all": len(ordered_tasks),
        "en": sum(1 for item in task_manifest["tasks"] if item["language"] == "en"),
        "zh": sum(1 for item in task_manifest["tasks"] if item["language"] == "zh"),
    }

    methods = [args.method_label]
    latest = collect_latest_runs(final_results_root, args.method_label)
    summary_payload = {
        "task_group": task_manifest.get("task_group"),
        "artifact_dir": str(artifact_dir),
        "benchmark_config": str(benchmark_config),
        "task_manifest": str(task_manifest_path),
        "methods": methods,
        "tasks": {},
    }
    for task_slug in ordered_tasks:
        summary_payload["tasks"][task_slug] = {}
        for method in methods:
            row = latest.get(task_slug)
            if row is None:
                continue
            summary_payload["tasks"][task_slug][method] = {
                "accuracy": row["accuracy"],
                "evaluated_count": row["evaluated_count"],
                "correct_count": row["correct_count"],
                "run_dir": row["run_dir"],
                "summary_path": row["summary_path"],
            }
    summary_payload["aggregates"] = build_aggregates(summary_payload["tasks"], task_language, methods)

    summary_json = summary_root / f"{args.summary_stem}.json"
    summary_md = summary_root / f"{args.summary_stem}.md"
    write_json(summary_json, summary_payload)
    summary_md.write_text(build_markdown(summary_payload, ordered_tasks, task_counts), encoding="utf-8")

    print(f"[OK] summary_json={summary_json}")
    print(f"[OK] summary_md={summary_md}")
    macro_all = summary_payload["aggregates"]["macro_average"][args.method_label]["all"]
    print(f"[OK] macro_average_all={macro_all}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
