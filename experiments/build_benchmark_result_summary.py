import json
from pathlib import Path


EXPERIMENTS_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = EXPERIMENTS_ROOT / "results"
SUMMARY_JSON = RESULTS_ROOT / "benchmark_result_summary.json"
SUMMARY_MD = RESULTS_ROOT / "benchmark_result_summary.md"
MANIFEST_JSON = EXPERIMENTS_ROOT / "data" / "benchmarks" / "benchmark_manifest.json"
BENCHMARK_DATA_ROOT = EXPERIMENTS_ROOT / "data" / "benchmarks"

METHOD_ORDER = ["baseline", "sparsegpt", "admm", "gptq", "awq", "smoothquant"]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_manifest_index():
    if not MANIFEST_JSON.exists():
        return {}
    manifest = read_json(MANIFEST_JSON)
    exports = manifest.get("exports", [])
    return {Path(item["output_path"]).stem: item for item in exports}


def load_task_metadata(task_slug: str, manifest_index: dict):
    manifest_row = manifest_index.get(task_slug)
    if manifest_row:
        return {
            "language": manifest_row.get("language", "unknown"),
            "row_count": manifest_row.get("row_count"),
        }

    data_path = BENCHMARK_DATA_ROOT / f"{task_slug}.jsonl"
    if not data_path.exists():
        return {"language": "unknown", "row_count": None}

    language = "unknown"
    row_count = 0
    with data_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            row_count += 1
            if language == "unknown":
                language = str(record.get("language", "unknown"))
    return {
        "language": language,
        "row_count": row_count,
    }


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


def build_aggregates(tasks_payload: dict, manifest_index: dict):
    aggregates = {
        "macro_average": {},
        "weighted_average": {},
        "task_counts": {
            "all": 0,
            "en": 0,
            "zh": 0,
        },
        "sample_counts": {
            "all": 0,
            "en": 0,
            "zh": 0,
        },
    }

    task_lang = {}
    task_row_count = {}
    for task_slug in tasks_payload:
        metadata = load_task_metadata(task_slug, manifest_index)
        language = metadata["language"]
        task_lang[task_slug] = language
        task_row_count[task_slug] = metadata["row_count"]
        aggregates["task_counts"]["all"] += 1
        if language in ("en", "zh"):
            aggregates["task_counts"][language] += 1

    for method in METHOD_ORDER:
        macro_vals = {"all": [], "en": [], "zh": []}
        weighted = {
            "all": {"correct": 0, "evaluated": 0},
            "en": {"correct": 0, "evaluated": 0},
            "zh": {"correct": 0, "evaluated": 0},
        }
        for task_slug, rows in tasks_payload.items():
            row = rows.get(method)
            if not row:
                continue
            language = task_lang.get(task_slug, "unknown")
            accuracy = row["accuracy"]
            correct = row["correct_count"]
            evaluated = row["evaluated_count"]
            macro_vals["all"].append(accuracy)
            weighted["all"]["correct"] += correct
            weighted["all"]["evaluated"] += evaluated
            if language in ("en", "zh"):
                macro_vals[language].append(accuracy)
                weighted[language]["correct"] += correct
                weighted[language]["evaluated"] += evaluated

        aggregates["macro_average"][method] = {}
        aggregates["weighted_average"][method] = {}
        for bucket in ("all", "en", "zh"):
            vals = macro_vals[bucket]
            aggregates["macro_average"][method][bucket] = (
                round(sum(vals) / len(vals), 6) if vals else None
            )
            total_eval = weighted[bucket]["evaluated"]
            aggregates["weighted_average"][method][bucket] = (
                round(weighted[bucket]["correct"] / total_eval, 6) if total_eval else None
            )

        aggregates["sample_counts"]["all"] = weighted["all"]["evaluated"]
        aggregates["sample_counts"]["en"] = weighted["en"]["evaluated"]
        aggregates["sample_counts"]["zh"] = weighted["zh"]["evaluated"]

    return aggregates


def main() -> int:
    latest = collect_latest_runs()
    manifest_index = load_manifest_index()
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

    payload["aggregates"] = build_aggregates(payload["tasks"], manifest_index)

    SUMMARY_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = ["# Benchmark Result Summary", ""]
    lines.append("## Aggregate Metrics")
    lines.append("")
    lines.append("Macro average is computed by giving each task equal weight within the selected task set.")
    lines.append("Weighted average is computed by weighting each task by its evaluated sample count.")
    lines.append("")
    task_counts = payload["aggregates"]["task_counts"]
    sample_counts = payload["aggregates"]["sample_counts"]
    lines.append(
        f"| method | macro(all {task_counts['all']}) | weighted(all {sample_counts['all']}) | "
        f"macro(en {task_counts['en']}) | weighted(en {sample_counts['en']}) | "
        f"macro(zh {task_counts['zh']}) | weighted(zh {sample_counts['zh']}) |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for method in METHOD_ORDER:
        macro = payload["aggregates"]["macro_average"][method]
        weighted = payload["aggregates"]["weighted_average"][method]
        lines.append(
            f"| {method} | {macro['all']} | {weighted['all']} | {macro['en']} | {weighted['en']} | {macro['zh']} | {weighted['zh']} |"
        )
    lines.append("")
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
