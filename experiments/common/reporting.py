import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def create_run_dir(output_root: Path, method: str, experiment_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = output_root / method / f"{timestamp}-{experiment_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def summarize_directory(path: Path) -> Dict[str, Any]:
    total_bytes = 0
    file_count = 0
    safetensors_count = 0
    json_count = 0
    other_count = 0
    largest_file = {"path": "", "size_bytes": 0}

    for child in path.rglob("*"):
        if not child.is_file():
            continue
        size_bytes = child.stat().st_size
        total_bytes += size_bytes
        file_count += 1
        suffix = child.suffix.lower()
        if suffix == ".safetensors":
            safetensors_count += 1
        elif suffix == ".json":
            json_count += 1
        else:
            other_count += 1
        if size_bytes > largest_file["size_bytes"]:
            largest_file = {
                "path": str(child.relative_to(path)),
                "size_bytes": size_bytes,
            }

    return {
        "path": str(path),
        "file_count": file_count,
        "total_size_bytes": total_bytes,
        "safetensors_file_count": safetensors_count,
        "json_file_count": json_count,
        "other_file_count": other_count,
        "largest_file": largest_file,
    }
