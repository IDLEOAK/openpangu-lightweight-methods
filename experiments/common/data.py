import json
from pathlib import Path
from typing import List, Optional


DEFAULT_TEXTS = [
    "请简要介绍大语言模型。",
    "Explain post-training pruning in one paragraph.",
    "Why is GPTQ useful for LLM deployment?",
]


def load_text_samples(data_path: Optional[Path], limit: int) -> List[str]:
    if data_path is None or not data_path.exists():
        return DEFAULT_TEXTS[:limit]

    if data_path.suffix.lower() == ".jsonl":
        samples = []
        with data_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                text = (
                    record.get("text")
                    or record.get("prompt")
                    or record.get("content")
                    or record.get("input")
                )
                if text:
                    samples.append(str(text))
                if len(samples) >= limit:
                    break
        return samples

    samples = []
    with data_path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            samples.append(text)
            if len(samples) >= limit:
                break
    return samples
