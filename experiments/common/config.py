import json
from pathlib import Path
from typing import Dict, Optional, Tuple


def load_config(config_path: str) -> Tuple[Dict, Path]:
    path = Path(config_path).resolve()
    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    return config, path


def resolve_path(base_dir: Path, value: Optional[str]) -> Optional[Path]:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path
