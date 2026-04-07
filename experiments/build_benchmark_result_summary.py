import json
from pathlib import Path


EXPERIMENTS_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = EXPERIMENTS_ROOT / "results"
SUMMARY_JSON = RESULTS_ROOT / "benchmark_result_summary.json"
SUMMARY_MD = RESULTS_ROOT / "benchmark_result_summary.md"

METHOD_ORDER = ["baseline", "sparsegpt", "admm", "gptq", "awq", "smoothquant"]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def extract_accuracy(summary: dict):
    if summary.get("method") == "benchmark":
        result = summary.get("benchmark_result", {})
        return "baseline", result.get("accuracy"), result.get("evaluated_count"), result.get("correct_count")

    for key, label in [
        ("pruned_benchmark", "sparsegpt"),
        ("quantized_benchmark", None),
    ]:
        if key in summary:
            method_label = summary.get("method")
            result = summary[key]
            return method_label, result.get("accuracy"), result.get("evaluated_count"), result.get("correct_count")
    return None, None, None, None


def collect_latest_runs():
    latest = {}
    for method_dir in RESULTS_ROOT.iterdir():
        if not method_dir.is_dir():
            continue
        for run_dir in method_dir.iterdir():
            summary_path = run_dir / "summary.json"
            if not summary_path.exists():
                continue
            summary = read_json(summary_path)
            benchmark_plan = summary.get("benchmark_plan")
            if summary.get("method") != "benchmark" and not benchmark_plan:
                continue
            task_slug = (
                benchmark_plan.get("task_slug")
                if benchmark_plan
                else Path(summary.get("benchmark_data_path", "")).stem
            )
            method_label, accuracy, evaluated_count, correct_count = extract_accuracy(summary)
            if not method_label or not task_slug:
                continue
            key = (method_label, task_slug)
            current = latest.get(key)
            mtime = summary_path.stat().st_mtime
            if current is None or mtime > current["mtime"]:
                latest[key] = {
                    "method": method_label,
                    "task_slug": task_slug,
                    "run_dir": str(run_dir),
                    "accuracy": accuracy,
                    "evaluated_count": evaluated_count,
                    "correct_count": correct_count,
                    "mtime": mtime,
                }
    return latest


def main() -> int:
    latest = collect_latest_runs()
    tasks = sorted({task_slug for _, task_slug in latest.keys()})

    payload = {"tasks": {}, "methods": METHOD_ORDER}
    for task_slug in tasks:
        payload["tasks"][task_slug] = {}
        for method in METHOD_ORDER:
            row = latest.get((method, task_slug))
            if row:
                payload["tasks"][task_slug][method] = {
                    "accuracy": row["accuracy"],
                    "evaluated_count": row["evaluated_count"],
                    "correct_count": row["correct_count"],
                    "run_dir": row["run_dir"],
                }

    SUMMARY_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = ["# Benchmark Result Summary", ""]
    for task_slug in tasks:
        lines.append(f"## {task_slug}")
        lines.append("")
        lines.append("| method | accuracy | correct/evaluated | run_dir |")
        lines.append("| --- | ---: | ---: | --- |")
        for method in METHOD_ORDER:
            row = payload["tasks"][task_slug].get(method)
            if not row:
                lines.append(f"| {method} | -- | -- | -- |")
                continue
            lines.append(
                f"| {method} | {row['accuracy']} | {row['correct_count']}/{row['evaluated_count']} | `{row['run_dir']}` |"
            )
        lines.append("")

    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] json={SUMMARY_JSON}")
    print(f"[OK] markdown={SUMMARY_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
