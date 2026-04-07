import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import torch

from experiments.common.config import resolve_path
from experiments.common.runtime import get_model_input_device


def load_multiple_choice_samples(data_path: Path, limit: int = 0) -> List[Dict]:
    samples = []
    with data_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "prompt" not in record or "choices" not in record or "answer_index" not in record:
                raise ValueError(f"Invalid benchmark record in {data_path}: missing prompt/choices/answer_index")
            record["choices"] = [str(choice) for choice in record["choices"]]
            record["answer_index"] = int(record["answer_index"])
            record.setdefault("task_name", data_path.stem)
            record.setdefault("sample_id", f"{record['task_name']}_{len(samples):06d}")
            samples.append(record)
            if limit and len(samples) >= limit:
                break
    return samples


def apply_benchmark_overrides(
    config: Dict,
    benchmark_data: str = "",
    benchmark_limit: Optional[int] = None,
    benchmark_max_length: Optional[int] = None,
    benchmark_scoring_mode: str = "",
) -> None:
    benchmark_cfg = config.setdefault("benchmark_data", {})
    benchmark_eval_cfg = config.setdefault("benchmark_evaluation", {})
    if benchmark_data:
        benchmark_cfg["path"] = benchmark_data
    if benchmark_limit is not None:
        benchmark_cfg["limit"] = benchmark_limit
    if benchmark_max_length is not None:
        benchmark_eval_cfg["max_length"] = benchmark_max_length
    if benchmark_scoring_mode:
        benchmark_eval_cfg["scoring_mode"] = benchmark_scoring_mode


def load_benchmark_plan(base_dir: Path, config: Dict) -> Optional[Dict]:
    benchmark_cfg = config.get("benchmark_data", {})
    benchmark_path = resolve_path(base_dir, benchmark_cfg.get("path"))
    if benchmark_path is None:
        return None

    samples = load_multiple_choice_samples(benchmark_path, int(benchmark_cfg.get("limit", 0)))
    benchmark_eval_cfg = config.get("benchmark_evaluation", {})
    return {
        "path": benchmark_path,
        "task_slug": benchmark_path.stem,
        "samples": samples,
        "limit": len(samples),
        "max_length": int(benchmark_eval_cfg.get("max_length", 1536)),
        "scoring_mode": benchmark_eval_cfg.get("scoring_mode", "avg_logprob"),
    }


def render_chat_prompt(tokenizer, system_prompt: str, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{system_prompt}\n{user_prompt}"


def score_choice(
    model,
    tokenizer,
    prompt_text: str,
    choice_text: str,
    runtime_device: torch.device,
    device_map: str,
    max_length: int,
) -> Optional[Dict]:
    input_device = get_model_input_device(model, runtime_device, device_map)

    prompt_encoded = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    full_encoded = tokenizer(
        prompt_text + choice_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )

    prompt_length = int(prompt_encoded["input_ids"].shape[1])
    input_ids = full_encoded["input_ids"].to(input_device)
    attention_mask = full_encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(input_device)

    if input_ids.shape[1] <= prompt_length:
        return None

    labels = input_ids.clone()
    labels[:, :prompt_length] = -100
    target_token_count = int((labels != -100).sum().item())
    if target_token_count <= 0:
        return None

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    avg_logprob = -float(outputs.loss.detach().float().cpu())
    total_logprob = avg_logprob * target_token_count
    return {
        "choice": choice_text,
        "avg_logprob": round(avg_logprob, 6),
        "total_logprob": round(total_logprob, 6),
        "token_count": target_token_count,
    }


def evaluate_multiple_choice(
    model,
    tokenizer,
    samples: List[Dict],
    system_prompt: str,
    runtime_device: torch.device,
    device_map: str,
    max_length: int,
    scoring_mode: str = "avg_logprob",
) -> Dict:
    if scoring_mode not in {"avg_logprob", "total_logprob"}:
        raise ValueError(f"Unsupported scoring_mode: {scoring_mode}")

    results = []
    task_stats = defaultdict(lambda: {"correct": 0, "sample_count": 0, "skipped": 0})

    for sample in samples:
        rendered_prompt = render_chat_prompt(tokenizer, system_prompt, str(sample["prompt"]))
        choice_scores = []
        for choice_index, choice in enumerate(sample["choices"]):
            score = score_choice(
                model,
                tokenizer,
                rendered_prompt,
                str(choice),
                runtime_device,
                device_map,
                max_length,
            )
            if score is None:
                continue
            score["choice_index"] = choice_index
            choice_scores.append(score)

        task_name = str(sample["task_name"])
        task_stats[task_name]["sample_count"] += 1

        if not choice_scores:
            task_stats[task_name]["skipped"] += 1
            results.append(
                {
                    "sample_id": sample["sample_id"],
                    "task_name": task_name,
                    "skipped": True,
                    "reason": "all choices truncated or invalid",
                }
            )
            continue

        best_choice = max(choice_scores, key=lambda item: item[scoring_mode])
        correct = int(best_choice["choice_index"]) == int(sample["answer_index"])
        if correct:
            task_stats[task_name]["correct"] += 1

        results.append(
            {
                "sample_id": sample["sample_id"],
                "task_name": task_name,
                "language": sample.get("language"),
                "source_dataset": sample.get("source_dataset"),
                "split": sample.get("split"),
                "answer_index": int(sample["answer_index"]),
                "predicted_index": int(best_choice["choice_index"]),
                "correct": correct,
                "scoring_mode": scoring_mode,
                "choices": sample["choices"],
                "choice_scores": choice_scores,
            }
        )

    aggregate_correct = sum(item["correct"] for item in task_stats.values())
    aggregate_samples = sum(item["sample_count"] for item in task_stats.values())
    aggregate_skipped = sum(item["skipped"] for item in task_stats.values())
    aggregate_evaluated = aggregate_samples - aggregate_skipped

    per_task = {}
    for task_name, stats in task_stats.items():
        evaluated = stats["sample_count"] - stats["skipped"]
        per_task[task_name] = {
            "sample_count": stats["sample_count"],
            "evaluated_count": evaluated,
            "skipped_count": stats["skipped"],
            "correct_count": stats["correct"],
            "accuracy": round(stats["correct"] / evaluated, 6) if evaluated else None,
        }

    return {
        "sample_count": aggregate_samples,
        "evaluated_count": aggregate_evaluated,
        "skipped_count": aggregate_skipped,
        "correct_count": aggregate_correct,
        "accuracy": round(aggregate_correct / aggregate_evaluated, 6) if aggregate_evaluated else None,
        "scoring_mode": scoring_mode,
        "tasks": per_task,
        "results": results,
    }
