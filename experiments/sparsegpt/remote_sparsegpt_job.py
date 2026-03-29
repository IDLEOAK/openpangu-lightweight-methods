import json
import subprocess
import sys
import tarfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple


# Remote Job Configuration
# Edit only this block on the server when needed.
MODEL_PATH = "/home/think4090/jlj/openpangu-embedded-7b-model"
HF_HOME = "/home/think4090/jlj/openpangu-embedded-7b-model/.hf_cache"
OUTPUT_DIR = "/home/think4090/jlj/openpangu-embedded-7b-model/experiments/results"
CONFIG_PATH = "/home/think4090/jlj/openpangu-embedded-7b-model/experiments/configs/sparsegpt_port_minimal.json"
MIN_LAYER = 0
MAX_LAYER = 1
SPARSITY = 0.3
CALIBRATION_LIMIT = 2
CALIBRATION_MAX_LENGTH = 128
EXPERIMENT_NAME_SUFFIX = "server_minimal"
SAVE_DIR = ""
ARCHIVE_RESULTS = True
ARCHIVE_PATH = "/home/think4090/jlj/openpangu-embedded-7b-model/experiments/results/sparsegpt/server_minimal_result.tar.gz"


def resolve_runtime_paths():
    root = Path(__file__).resolve().parents[2]
    model_path = Path(MODEL_PATH).resolve() if MODEL_PATH else root
    hf_home = Path(HF_HOME).resolve() if HF_HOME else root / ".hf_cache"
    output_dir = Path(OUTPUT_DIR).resolve() if OUTPUT_DIR else root / "experiments" / "results"
    config_path = Path(CONFIG_PATH).resolve() if CONFIG_PATH else root / "experiments" / "configs" / "sparsegpt_port_minimal.json"
    save_dir = Path(SAVE_DIR).resolve() if SAVE_DIR else None
    archive_path = Path(ARCHIVE_PATH).resolve() if ARCHIVE_PATH else None
    return root, model_path, hf_home, output_dir, config_path, save_dir, archive_path


def build_command(root: Path, model_path: Path, hf_home: Path, output_dir: Path, config_path: Path, save_dir: Optional[Path]):
    cmd = [
        sys.executable,
        str(root / "experiments" / "sparsegpt" / "run_sparsegpt_scaffold.py"),
        "--config",
        str(config_path),
        "--model-path",
        str(model_path),
        "--hf-home",
        str(hf_home),
        "--output-dir",
        str(output_dir),
        "--min-layer",
        str(MIN_LAYER),
        "--max-layer",
        str(MAX_LAYER),
        "--sparsity",
        str(SPARSITY),
        "--calibration-limit",
        str(CALIBRATION_LIMIT),
        "--calibration-max-length",
        str(CALIBRATION_MAX_LENGTH),
        "--experiment-name-suffix",
        EXPERIMENT_NAME_SUFFIX,
    ]
    if save_dir is not None:
        cmd.extend(["--save-dir", str(save_dir)])
    return cmd


def find_latest_run_dir(output_dir: Path) -> Path:
    sparse_root = output_dir / "sparsegpt"
    if not sparse_root.exists():
        raise FileNotFoundError(f"SparseGPT result root not found: {sparse_root}")

    candidates = []
    for child in sparse_root.iterdir():
        if not child.is_dir():
            continue
        summary_path = child / "summary.json"
        if not summary_path.exists():
            continue
        candidates.append(child)

    if not candidates:
        raise FileNotFoundError(f"No completed SparseGPT run directory found under: {sparse_root}")

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def package_run_dir(run_dir: Path, archive_path: Optional[Path]) -> Path:
    if archive_path is None:
        archive_path = run_dir.parent / f"{run_dir.name}.tar.gz"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(run_dir, arcname=run_dir.name)
    return archive_path


def write_job_manifest(run_dir: Path, archive_path: Optional[Path], cmd: List[str]) -> Path:
    manifest = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python": sys.executable,
        "command": cmd,
        "run_dir": str(run_dir),
        "archive_path": str(archive_path) if archive_path else "",
        "config": {
            "min_layer": MIN_LAYER,
            "max_layer": MAX_LAYER,
            "sparsity": SPARSITY,
            "calibration_limit": CALIBRATION_LIMIT,
            "calibration_max_length": CALIBRATION_MAX_LENGTH,
            "experiment_name_suffix": EXPERIMENT_NAME_SUFFIX,
            "archive_results": ARCHIVE_RESULTS,
        },
    }
    manifest_path = run_dir / "remote_job_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def main() -> int:
    root, model_path, hf_home, output_dir, config_path, save_dir, archive_path = resolve_runtime_paths()
    hf_home.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_command(root, model_path, hf_home, output_dir, config_path, save_dir)

    print(f"[INFO] ROOT={root}")
    print(f"[INFO] MODEL_PATH={model_path}")
    print(f"[INFO] HF_HOME={hf_home}")
    print(f"[INFO] OUTPUT_DIR={output_dir}")
    print(f"[INFO] CONFIG_PATH={config_path}")
    print(f"[INFO] MIN_LAYER={MIN_LAYER} MAX_LAYER={MAX_LAYER}")
    print(f"[INFO] SPARSITY={SPARSITY}")
    print(f"[INFO] CALIBRATION_LIMIT={CALIBRATION_LIMIT} CALIBRATION_MAX_LENGTH={CALIBRATION_MAX_LENGTH}")
    print(f"[INFO] EXPERIMENT_NAME_SUFFIX={EXPERIMENT_NAME_SUFFIX}")
    print(f"[INFO] PYTHON={sys.executable}")

    completed = subprocess.run(cmd, cwd=root)
    if completed.returncode != 0:
        return completed.returncode

    run_dir = find_latest_run_dir(output_dir)
    created_archive = None
    if ARCHIVE_RESULTS:
        created_archive = package_run_dir(run_dir, archive_path)
        print(f"[OK] archive={created_archive}")

    manifest_path = write_job_manifest(run_dir, created_archive, cmd)
    print(f"[OK] run_dir={run_dir}")
    print(f"[OK] manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
