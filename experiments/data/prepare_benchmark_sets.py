import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List


TASK_REGISTRY: Dict[str, Dict[str, str]] = {
    "boolq": {
        "dataset": "boolq",
        "config": "",
        "split": "validation",
        "language": "en",
        "filename": "boolq_validation_mcq.jsonl",
    },
    "hellaswag": {
        "dataset": "hellaswag",
        "config": "",
        "split": "validation",
        "language": "en",
        "filename": "hellaswag_validation_mcq.jsonl",
    },
    "mmlu_college_computer_science": {
        "dataset": "cais/mmlu",
        "config": "college_computer_science",
        "split": "validation",
        "language": "en",
        "filename": "mmlu_college_computer_science_validation_mcq.jsonl",
    },
    "mmlu_high_school_computer_science": {
        "dataset": "cais/mmlu",
        "config": "high_school_computer_science",
        "split": "validation",
        "language": "en",
        "filename": "mmlu_high_school_computer_science_validation_mcq.jsonl",
    },
    "mmlu_computer_security": {
        "dataset": "cais/mmlu",
        "config": "computer_security",
        "split": "validation",
        "language": "en",
        "filename": "mmlu_computer_security_validation_mcq.jsonl",
    },
    "mmlu_machine_learning": {
        "dataset": "cais/mmlu",
        "config": "machine_learning",
        "split": "validation",
        "language": "en",
        "filename": "mmlu_machine_learning_validation_mcq.jsonl",
    },
    "mmlu_formal_logic": {
        "dataset": "cais/mmlu",
        "config": "formal_logic",
        "split": "validation",
        "language": "en",
        "filename": "mmlu_formal_logic_validation_mcq.jsonl",
    },
    "mmlu_college_mathematics": {
        "dataset": "cais/mmlu",
        "config": "college_mathematics",
        "split": "validation",
        "language": "en",
        "filename": "mmlu_college_mathematics_validation_mcq.jsonl",
    },
    "cmmlu_computer_science": {
        "dataset": "svjack/cmmlu",
        "config": "",
        "split": "train",
        "language": "zh",
        "filename": "cmmlu_computer_science_train_mcq.jsonl",
        "subject_key": "task",
        "subject_value": "computer_science",
    },
    "cmmlu_computer_security": {
        "dataset": "svjack/cmmlu",
        "config": "",
        "split": "train",
        "language": "zh",
        "filename": "cmmlu_computer_security_train_mcq.jsonl",
        "subject_key": "task",
        "subject_value": "computer_security",
    },
    "cmmlu_machine_learning": {
        "dataset": "svjack/cmmlu",
        "config": "",
        "split": "train",
        "language": "zh",
        "filename": "cmmlu_machine_learning_train_mcq.jsonl",
        "subject_key": "task",
        "subject_value": "machine_learning",
    },
    "cmmlu_high_school_mathematics": {
        "dataset": "svjack/cmmlu",
        "config": "",
        "split": "train",
        "language": "zh",
        "filename": "cmmlu_high_school_mathematics_train_mcq.jsonl",
        "subject_key": "task",
        "subject_value": "high_school_mathematics",
    },
    "cmmlu_college_mathematics": {
        "dataset": "svjack/cmmlu",
        "config": "",
        "split": "train",
        "language": "zh",
        "filename": "cmmlu_college_mathematics_train_mcq.jsonl",
        "subject_key": "task",
        "subject_value": "college_mathematics",
    },
    "ceval_college_programming": {
        "dataset": "ceval/ceval-exam",
        "config": "college_programming",
        "split": "val",
        "language": "zh",
        "filename": "ceval_college_programming_val_mcq.jsonl",
    },
    "ceval_computer_network": {
        "dataset": "ceval/ceval-exam",
        "config": "computer_network",
        "split": "val",
        "language": "zh",
        "filename": "ceval_computer_network_val_mcq.jsonl",
    },
    "ceval_operating_system": {
        "dataset": "ceval/ceval-exam",
        "config": "operating_system",
        "split": "val",
        "language": "zh",
        "filename": "ceval_operating_system_val_mcq.jsonl",
    },
    "ceval_computer_architecture": {
        "dataset": "ceval/ceval-exam",
        "config": "computer_architecture",
        "split": "val",
        "language": "zh",
        "filename": "ceval_computer_architecture_val_mcq.jsonl",
    },
}


TASK_GROUPS: Dict[str, List[str]] = {
    "smoke": [
        "boolq",
        "hellaswag",
        "mmlu_college_computer_science",
        "ceval_college_programming",
        "ceval_computer_network",
    ],
    "mmlu_core_en": [
        "mmlu_college_computer_science",
        "mmlu_high_school_computer_science",
        "mmlu_computer_security",
        "mmlu_machine_learning",
        "mmlu_formal_logic",
        "mmlu_college_mathematics",
    ],
    "cmmlu_core_zh": [
        "cmmlu_computer_science",
        "cmmlu_computer_security",
        "cmmlu_machine_learning",
        "cmmlu_high_school_mathematics",
        "cmmlu_college_mathematics",
    ],
    "ceval_core_zh": [
        "ceval_college_programming",
        "ceval_computer_network",
        "ceval_operating_system",
        "ceval_computer_architecture",
    ],
    "stage_c_core": [
        "ceval_college_programming",
        "ceval_computer_network",
        "ceval_operating_system",
        "ceval_computer_architecture",
        "mmlu_college_computer_science",
        "mmlu_computer_security",
        "mmlu_machine_learning",
        "mmlu_formal_logic",
    ],
    "formal_core": [
        "boolq",
        "hellaswag",
        "mmlu_college_computer_science",
        "mmlu_high_school_computer_science",
        "mmlu_computer_security",
        "mmlu_machine_learning",
        "mmlu_formal_logic",
        "mmlu_college_mathematics",
        "cmmlu_computer_science",
        "cmmlu_computer_security",
        "cmmlu_machine_learning",
        "cmmlu_high_school_mathematics",
        "cmmlu_college_mathematics",
        "ceval_college_programming",
        "ceval_computer_network",
    ],
}


def _load_dataset(task_name: str):
    from datasets import load_dataset

    task = TASK_REGISTRY[task_name]
    config_name = task["config"] or None
    return load_dataset(task["dataset"], config_name, split=task["split"])


def _write_records(output_path: Path, records: Iterable[Dict]) -> int:
    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def _build_boolq_record(item: Dict, idx: int) -> Dict:
    prompt = (
        "Read the passage and answer the question by selecting the correct option.\n\n"
        f"Passage: {str(item['passage']).strip()}\n"
        f"Question: {str(item['question']).strip()}\n"
        "Options:\n"
        "A. Yes\n"
        "B. No\n"
        "Answer:"
    )
    return {
        "sample_id": f"boolq_{idx:06d}",
        "task_name": "boolq",
        "language": "en",
        "source_dataset": "boolq",
        "split": "validation",
        "prompt": prompt,
        "choices": [" A", " B"],
        "answer_index": 0 if bool(item["answer"]) else 1,
    }


def _build_hellaswag_record(item: Dict, idx: int) -> Dict:
    prompt = (
        "Choose the best ending for the following scenario.\n\n"
        f"Scenario: {str(item['ctx']).strip()}\n"
        "Options:\n"
        f"A. {str(item['endings'][0]).strip()}\n"
        f"B. {str(item['endings'][1]).strip()}\n"
        f"C. {str(item['endings'][2]).strip()}\n"
        f"D. {str(item['endings'][3]).strip()}\n"
        "Answer:"
    )
    return {
        "sample_id": f"hellaswag_{idx:06d}",
        "task_name": "hellaswag",
        "language": "en",
        "source_dataset": "hellaswag",
        "split": "validation",
        "prompt": prompt,
        "choices": [" A", " B", " C", " D"],
        "answer_index": int(item["label"]),
    }


def _build_mmlu_record(item: Dict, idx: int, config_name: str) -> Dict:
    prompt = (
        "Answer the following multiple-choice question by selecting the correct option.\n\n"
        f"Question: {str(item['question']).strip()}\n"
        "Options:\n"
        f"A. {str(item['choices'][0]).strip()}\n"
        f"B. {str(item['choices'][1]).strip()}\n"
        f"C. {str(item['choices'][2]).strip()}\n"
        f"D. {str(item['choices'][3]).strip()}\n"
        "Answer:"
    )
    return {
        "sample_id": f"mmlu_{config_name}_{idx:06d}",
        "task_name": f"mmlu_{config_name}",
        "language": "en",
        "source_dataset": "cais/mmlu",
        "split": "validation",
        "prompt": prompt,
        "choices": [" A", " B", " C", " D"],
        "answer_index": int(item["answer"]),
    }


def _build_cmmlu_record(item: Dict, idx: int, subject_value: str) -> Dict:
    answer_map = {"A": 0, "B": 1, "C": 2, "D": 3}
    prompt = (
        "请阅读题目并选择正确选项。\n\n"
        f"题目：{str(item['question']).strip()}\n"
        "选项：\n"
        f"A. {str(item['A']).strip()}\n"
        f"B. {str(item['B']).strip()}\n"
        f"C. {str(item['C']).strip()}\n"
        f"D. {str(item['D']).strip()}\n"
        "答案："
    )
    return {
        "sample_id": f"cmmlu_{subject_value}_{idx:06d}",
        "task_name": f"cmmlu_{subject_value}",
        "language": "zh",
        "source_dataset": "svjack/cmmlu",
        "split": "train",
        "prompt": prompt,
        "choices": ["A", "B", "C", "D"],
        "answer_index": answer_map[str(item["answer"]).strip()],
    }


def _build_ceval_record(item: Dict, idx: int, config_name: str) -> Dict:
    answer_map = {"A": 0, "B": 1, "C": 2, "D": 3}
    prompt = (
        "请阅读题目并选择正确选项。\n\n"
        f"题目：{str(item['question']).strip()}\n"
        "选项：\n"
        f"A. {str(item['A']).strip()}\n"
        f"B. {str(item['B']).strip()}\n"
        f"C. {str(item['C']).strip()}\n"
        f"D. {str(item['D']).strip()}\n"
        "答案："
    )
    return {
        "sample_id": f"ceval_{config_name}_{idx:06d}",
        "task_name": f"ceval_{config_name}",
        "language": "zh",
        "source_dataset": "ceval/ceval-exam",
        "split": "val",
        "prompt": prompt,
        "choices": ["A", "B", "C", "D"],
        "answer_index": answer_map[str(item["answer"]).strip()],
    }


def export_task(task_name: str, output_dir: Path, max_samples: int) -> Dict:
    dataset = _load_dataset(task_name)
    task = TASK_REGISTRY[task_name]
    config_name = task["config"]
    output_path = output_dir / task["filename"]
    subject_key = task.get("subject_key", "")
    subject_value = task.get("subject_value", "")

    records: List[Dict] = []
    kept_rows = 0
    for item in dataset:
        if subject_key and str(item.get(subject_key, "")).strip() != subject_value:
            continue

        if task_name == "boolq":
            record = _build_boolq_record(item, kept_rows)
        elif task_name == "hellaswag":
            record = _build_hellaswag_record(item, kept_rows)
        elif task_name.startswith("mmlu_"):
            record = _build_mmlu_record(item, kept_rows, config_name)
        elif task_name.startswith("cmmlu_"):
            record = _build_cmmlu_record(item, kept_rows, subject_value)
        elif task_name.startswith("ceval_"):
            record = _build_ceval_record(item, kept_rows, config_name)
        else:
            raise ValueError(f"Unsupported task: {task_name}")

        records.append(record)
        kept_rows += 1
        if max_samples and kept_rows >= max_samples:
            break

    row_count = _write_records(output_path, records)
    return {
        "task_name": task_name,
        "source_dataset": task["dataset"],
        "config": config_name,
        "split": task["split"],
        "language": task["language"],
        "subject_key": subject_key,
        "subject_value": subject_value,
        "output_path": str(output_path),
        "row_count": row_count,
    }


def parse_tasks(task_names: str) -> List[str]:
    normalized = (task_names or "").strip().lower()
    if not normalized or normalized == "default":
        return TASK_GROUPS["formal_core"]
    if normalized in TASK_GROUPS:
        return TASK_GROUPS[normalized]

    tasks = [name.strip() for name in task_names.split(",") if name.strip()]
    invalid = [name for name in tasks if name not in TASK_REGISTRY]
    if invalid:
        raise ValueError(f"Unknown benchmark task(s): {', '.join(invalid)}")
    return tasks


def main() -> int:
    parser = argparse.ArgumentParser(description="Export public multiple-choice benchmark sets to local jsonl files.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/data/benchmarks",
        help="Directory where exported benchmark jsonl files will be written.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="default",
        help="Task group name or comma-separated task names. Default is the formal_core task set.",
    )
    parser.add_argument(
        "--max-samples-per-task",
        type=int,
        default=128,
        help="Maximum number of samples exported for each task. Use 0 for all samples.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_tasks = parse_tasks(args.tasks)
    manifest = {
        "task_group": args.tasks,
        "task_names": selected_tasks,
        "max_samples_per_task": int(args.max_samples_per_task),
        "exports": [],
    }

    for task_name in selected_tasks:
        export_info = export_task(task_name, output_dir, int(args.max_samples_per_task))
        manifest["exports"].append(export_info)
        print(f"[OK] exported {export_info['row_count']} rows -> {export_info['output_path']}")

    manifest_path = output_dir / "benchmark_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
