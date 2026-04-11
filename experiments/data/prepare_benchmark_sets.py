import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional


CHOICE_LETTERS = ["A", "B", "C", "D"]
MMLU_FEW_SHOT_COUNT = 5
CEVAL_FEW_SHOT_COUNT = 5

MMLU_HARD_EN_TASKS = [
    "mmlu_abstract_algebra",
    "mmlu_college_mathematics",
    "mmlu_high_school_statistics",
    "mmlu_college_chemistry",
    "mmlu_college_physics",
    "mmlu_high_school_mathematics",
    "mmlu_high_school_chemistry",
    "mmlu_high_school_physics",
]

CEVAL_HARD_ZH_TASKS = [
    "ceval_advanced_mathematics",
    "ceval_discrete_mathematics",
    "ceval_probability_and_statistics",
    "ceval_college_chemistry",
    "ceval_college_physics",
    "ceval_high_school_mathematics",
    "ceval_high_school_chemistry",
    "ceval_high_school_physics",
]

FINAL_ARTIFACT_CORE_LEGACY_TASKS = [
    "mmlu_college_computer_science",
    "mmlu_computer_security",
    "mmlu_machine_learning",
    "mmlu_formal_logic",
    "ceval_college_programming",
    "ceval_computer_network",
    "ceval_operating_system",
    "ceval_computer_architecture",
]

CEVAL_SUBJECT_TITLES = {
    "advanced_mathematics": "高等数学",
    "college_architecture": "计算机组成",
    "college_chemistry": "大学化学",
    "college_physics": "大学物理",
    "college_programming": "大学编程",
    "computer_architecture": "计算机组成",
    "computer_network": "计算机网络",
    "discrete_mathematics": "离散数学",
    "high_school_chemistry": "高中化学",
    "high_school_mathematics": "高中数学",
    "high_school_physics": "高中物理",
    "operating_system": "操作系统",
    "probability_and_statistics": "概率统计",
}

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
        "dev_split": "dev",
        "language": "en",
        "filename": "mmlu_college_computer_science_validation_mcq.jsonl",
    },
    "mmlu_high_school_computer_science": {
        "dataset": "cais/mmlu",
        "config": "high_school_computer_science",
        "split": "validation",
        "dev_split": "dev",
        "language": "en",
        "filename": "mmlu_high_school_computer_science_validation_mcq.jsonl",
    },
    "mmlu_computer_security": {
        "dataset": "cais/mmlu",
        "config": "computer_security",
        "split": "validation",
        "dev_split": "dev",
        "language": "en",
        "filename": "mmlu_computer_security_validation_mcq.jsonl",
    },
    "mmlu_machine_learning": {
        "dataset": "cais/mmlu",
        "config": "machine_learning",
        "split": "validation",
        "dev_split": "dev",
        "language": "en",
        "filename": "mmlu_machine_learning_validation_mcq.jsonl",
    },
    "mmlu_formal_logic": {
        "dataset": "cais/mmlu",
        "config": "formal_logic",
        "split": "validation",
        "dev_split": "dev",
        "language": "en",
        "filename": "mmlu_formal_logic_validation_mcq.jsonl",
    },
    "mmlu_college_mathematics": {
        "dataset": "cais/mmlu",
        "config": "college_mathematics",
        "split": "validation",
        "dev_split": "dev",
        "language": "en",
        "filename": "mmlu_college_mathematics_validation_mcq.jsonl",
    },
    "mmlu_abstract_algebra": {
        "dataset": "cais/mmlu",
        "config": "abstract_algebra",
        "split": "validation",
        "dev_split": "dev",
        "language": "en",
        "filename": "mmlu_abstract_algebra_validation_mcq.jsonl",
    },
    "mmlu_high_school_statistics": {
        "dataset": "cais/mmlu",
        "config": "high_school_statistics",
        "split": "validation",
        "dev_split": "dev",
        "language": "en",
        "filename": "mmlu_high_school_statistics_validation_mcq.jsonl",
    },
    "mmlu_college_chemistry": {
        "dataset": "cais/mmlu",
        "config": "college_chemistry",
        "split": "validation",
        "dev_split": "dev",
        "language": "en",
        "filename": "mmlu_college_chemistry_validation_mcq.jsonl",
    },
    "mmlu_college_physics": {
        "dataset": "cais/mmlu",
        "config": "college_physics",
        "split": "validation",
        "dev_split": "dev",
        "language": "en",
        "filename": "mmlu_college_physics_validation_mcq.jsonl",
    },
    "mmlu_high_school_mathematics": {
        "dataset": "cais/mmlu",
        "config": "high_school_mathematics",
        "split": "validation",
        "dev_split": "dev",
        "language": "en",
        "filename": "mmlu_high_school_mathematics_validation_mcq.jsonl",
    },
    "mmlu_high_school_chemistry": {
        "dataset": "cais/mmlu",
        "config": "high_school_chemistry",
        "split": "validation",
        "dev_split": "dev",
        "language": "en",
        "filename": "mmlu_high_school_chemistry_validation_mcq.jsonl",
    },
    "mmlu_high_school_physics": {
        "dataset": "cais/mmlu",
        "config": "high_school_physics",
        "split": "validation",
        "dev_split": "dev",
        "language": "en",
        "filename": "mmlu_high_school_physics_validation_mcq.jsonl",
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
        "dev_split": "dev",
        "language": "zh",
        "filename": "ceval_college_programming_val_mcq.jsonl",
    },
    "ceval_computer_network": {
        "dataset": "ceval/ceval-exam",
        "config": "computer_network",
        "split": "val",
        "dev_split": "dev",
        "language": "zh",
        "filename": "ceval_computer_network_val_mcq.jsonl",
    },
    "ceval_operating_system": {
        "dataset": "ceval/ceval-exam",
        "config": "operating_system",
        "split": "val",
        "dev_split": "dev",
        "language": "zh",
        "filename": "ceval_operating_system_val_mcq.jsonl",
    },
    "ceval_computer_architecture": {
        "dataset": "ceval/ceval-exam",
        "config": "computer_architecture",
        "split": "val",
        "dev_split": "dev",
        "language": "zh",
        "filename": "ceval_computer_architecture_val_mcq.jsonl",
    },
    "ceval_advanced_mathematics": {
        "dataset": "ceval/ceval-exam",
        "config": "advanced_mathematics",
        "split": "val",
        "dev_split": "dev",
        "language": "zh",
        "filename": "ceval_advanced_mathematics_val_mcq.jsonl",
    },
    "ceval_discrete_mathematics": {
        "dataset": "ceval/ceval-exam",
        "config": "discrete_mathematics",
        "split": "val",
        "dev_split": "dev",
        "language": "zh",
        "filename": "ceval_discrete_mathematics_val_mcq.jsonl",
    },
    "ceval_probability_and_statistics": {
        "dataset": "ceval/ceval-exam",
        "config": "probability_and_statistics",
        "split": "val",
        "dev_split": "dev",
        "language": "zh",
        "filename": "ceval_probability_and_statistics_val_mcq.jsonl",
    },
    "ceval_college_chemistry": {
        "dataset": "ceval/ceval-exam",
        "config": "college_chemistry",
        "split": "val",
        "dev_split": "dev",
        "language": "zh",
        "filename": "ceval_college_chemistry_val_mcq.jsonl",
    },
    "ceval_college_physics": {
        "dataset": "ceval/ceval-exam",
        "config": "college_physics",
        "split": "val",
        "dev_split": "dev",
        "language": "zh",
        "filename": "ceval_college_physics_val_mcq.jsonl",
    },
    "ceval_high_school_mathematics": {
        "dataset": "ceval/ceval-exam",
        "config": "high_school_mathematics",
        "split": "val",
        "dev_split": "dev",
        "language": "zh",
        "filename": "ceval_high_school_mathematics_val_mcq.jsonl",
    },
    "ceval_high_school_chemistry": {
        "dataset": "ceval/ceval-exam",
        "config": "high_school_chemistry",
        "split": "val",
        "dev_split": "dev",
        "language": "zh",
        "filename": "ceval_high_school_chemistry_val_mcq.jsonl",
    },
    "ceval_high_school_physics": {
        "dataset": "ceval/ceval-exam",
        "config": "high_school_physics",
        "split": "val",
        "dev_split": "dev",
        "language": "zh",
        "filename": "ceval_high_school_physics_val_mcq.jsonl",
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
    "hard_smoke": [
        "mmlu_abstract_algebra",
        "ceval_advanced_mathematics",
    ],
    "mmlu_core_en": [
        "mmlu_college_computer_science",
        "mmlu_high_school_computer_science",
        "mmlu_computer_security",
        "mmlu_machine_learning",
        "mmlu_formal_logic",
        "mmlu_college_mathematics",
    ],
    "mmlu_hard_en": MMLU_HARD_EN_TASKS,
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
    "ceval_hard_zh": CEVAL_HARD_ZH_TASKS,
    "final_artifact_core": FINAL_ARTIFACT_CORE_LEGACY_TASKS,
    "final_artifact_core_legacy": FINAL_ARTIFACT_CORE_LEGACY_TASKS,
    "final_artifact_hard_8x8": MMLU_HARD_EN_TASKS + CEVAL_HARD_ZH_TASKS,
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
    "formal_hard_16": MMLU_HARD_EN_TASKS + CEVAL_HARD_ZH_TASKS,
}


def _load_dataset(task_name: str, split_override: Optional[str] = None):
    from datasets import load_dataset

    task = TASK_REGISTRY[task_name]
    config_name = task["config"] or None
    return load_dataset(task["dataset"], config_name, split=split_override or task["split"])


def _write_records(output_path: Path, records: Iterable[Dict]) -> int:
    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def _default_metadata(task_name: str, prompt_style: str, apply_chat_template: bool) -> Dict:
    return {
        "benchmark_family": task_name.split("_", 1)[0],
        "prompt_style": prompt_style,
        "apply_chat_template": apply_chat_template,
    }


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
        **_default_metadata("boolq", "custom_zero_shot", True),
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
        **_default_metadata("hellaswag", "custom_zero_shot", True),
    }


def _format_mmlu_subject(config_name: str) -> str:
    return config_name.replace("_", " ")


def _build_mmlu_example(item: Dict, include_answer: bool) -> str:
    example = str(item["question"]).strip()
    for choice_index, choice_text in enumerate(item["choices"]):
        example += f"\n{CHOICE_LETTERS[choice_index]}. {str(choice_text).strip()}"
    example += "\nAnswer:"
    if include_answer:
        example += f" {CHOICE_LETTERS[int(item['answer'])]}\n\n"
    return example


def _build_mmlu_prompt(config_name: str, dev_examples: List[Dict], item: Dict) -> str:
    prompt = (
        "The following are multiple choice questions (with answers) about "
        f"{_format_mmlu_subject(config_name)}.\n\n"
    )
    for example in dev_examples[:MMLU_FEW_SHOT_COUNT]:
        prompt += _build_mmlu_example(example, include_answer=True)
    prompt += _build_mmlu_example(item, include_answer=False)
    return prompt


def _build_mmlu_record(item: Dict, idx: int, config_name: str, dev_examples: List[Dict]) -> Dict:
    return {
        "sample_id": f"mmlu_{config_name}_{idx:06d}",
        "task_name": f"mmlu_{config_name}",
        "language": "en",
        "source_dataset": "cais/mmlu",
        "split": "validation",
        "prompt": _build_mmlu_prompt(config_name, dev_examples, item),
        "choices": [" A", " B", " C", " D"],
        "answer_index": int(item["answer"]),
        "few_shot_count": MMLU_FEW_SHOT_COUNT,
        "prompt_template_source": "hendrycks/test:evaluate.py",
        "subject_name": _format_mmlu_subject(config_name),
        **_default_metadata("mmlu", "official_few_shot", False),
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
        **_default_metadata("cmmlu", "custom_zero_shot", True),
    }


def _build_ceval_example(item: Dict, include_answer: bool) -> str:
    example = str(item["question"]).strip()
    for choice in CHOICE_LETTERS:
        example += f"\n{choice}. {str(item[choice]).strip()}"
    example += "\n答案："
    if include_answer:
        example += f"{str(item['answer']).strip()}\n\n"
    return example


def _build_ceval_prompt(subject_title: str, dev_examples: List[Dict], item: Dict) -> str:
    prompt = f"以下是中国关于{subject_title}考试的单项选择题，请选出其中的正确答案。\n\n"
    for example in dev_examples[:CEVAL_FEW_SHOT_COUNT]:
        prompt += _build_ceval_example(example, include_answer=True)
    prompt += _build_ceval_example(item, include_answer=False)
    return prompt


def _build_ceval_record(
    item: Dict,
    idx: int,
    config_name: str,
    subject_title: str,
    dev_examples: List[Dict],
) -> Dict:
    answer_map = {"A": 0, "B": 1, "C": 2, "D": 3}
    return {
        "sample_id": f"ceval_{config_name}_{idx:06d}",
        "task_name": f"ceval_{config_name}",
        "language": "zh",
        "source_dataset": "ceval/ceval-exam",
        "split": "val",
        "prompt": _build_ceval_prompt(subject_title, dev_examples, item),
        "choices": ["A", "B", "C", "D"],
        "answer_index": answer_map[str(item["answer"]).strip()],
        "few_shot_count": CEVAL_FEW_SHOT_COUNT,
        "prompt_template_source": "hkust-nlp/ceval:evaluators/evaluator.py",
        "subject_name": subject_title,
        **_default_metadata("ceval", "official_few_shot", False),
    }


def export_task(task_name: str, output_dir: Path, max_samples: int) -> Dict:
    dataset = _load_dataset(task_name)
    task = TASK_REGISTRY[task_name]
    config_name = task["config"]
    output_path = output_dir / task["filename"]
    subject_key = task.get("subject_key", "")
    subject_value = task.get("subject_value", "")
    dev_examples: List[Dict] = []
    if task_name.startswith("mmlu_") or task_name.startswith("ceval_"):
        dev_split = task.get("dev_split", "")
        if not dev_split:
            raise ValueError(f"Missing dev_split for task={task_name}")
        dev_examples = list(_load_dataset(task_name, split_override=dev_split))

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
            record = _build_mmlu_record(item, kept_rows, config_name, dev_examples)
        elif task_name.startswith("cmmlu_"):
            record = _build_cmmlu_record(item, kept_rows, subject_value)
        elif task_name.startswith("ceval_"):
            subject_title = CEVAL_SUBJECT_TITLES.get(config_name, config_name.replace("_", " "))
            record = _build_ceval_record(item, kept_rows, config_name, subject_title, dev_examples)
        else:
            raise ValueError(f"Unsupported task: {task_name}")

        records.append(record)
        kept_rows += 1
        if max_samples and kept_rows >= max_samples:
            break

    row_count = _write_records(output_path, records)
    export_info = {
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
    if dev_examples:
        export_info["prompt_style"] = "official_few_shot"
        export_info["few_shot_count"] = len(dev_examples[: MMLU_FEW_SHOT_COUNT if task_name.startswith("mmlu_") else CEVAL_FEW_SHOT_COUNT])
        export_info["apply_chat_template"] = False
    return export_info


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
