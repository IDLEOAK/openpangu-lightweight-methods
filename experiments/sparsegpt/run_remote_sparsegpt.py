import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(description="Python entrypoint for remote SparseGPT runs.")
    parser.add_argument("--model-path", default=str(root), help="Remote OpenPangu model directory.")
    parser.add_argument("--hf-home", default=str(root / ".hf_cache"), help="HF cache directory.")
    parser.add_argument("--output-dir", default=str(root / "experiments" / "results"), help="Result output directory.")
    parser.add_argument(
        "--config-path",
        default=str(root / "experiments" / "configs" / "sparsegpt_port_minimal.json"),
        help="Base config path.",
    )
    parser.add_argument("--min-layer", type=int, default=0, help="SparseGPT min layer override.")
    parser.add_argument("--max-layer", type=int, default=1, help="SparseGPT max layer override.")
    parser.add_argument("--sparsity", type=float, default=0.3, help="SparseGPT sparsity override.")
    parser.add_argument("--calibration-limit", type=int, default=2, help="Calibration sample limit.")
    parser.add_argument("--calibration-max-length", type=int, default=128, help="Calibration max token length.")
    parser.add_argument(
        "--experiment-name-suffix",
        default="remote",
        help="Suffix appended to the experiment name in the generated run directory.",
    )
    parser.add_argument("--save-dir", default="", help="Optional model save directory.")
    args = parser.parse_args()

    Path(args.hf_home).mkdir(parents=True, exist_ok=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(root / "experiments" / "sparsegpt" / "run_sparsegpt_scaffold.py"),
        "--config",
        args.config_path,
        "--model-path",
        args.model_path,
        "--hf-home",
        args.hf_home,
        "--output-dir",
        args.output_dir,
        "--min-layer",
        str(args.min_layer),
        "--max-layer",
        str(args.max_layer),
        "--sparsity",
        str(args.sparsity),
        "--calibration-limit",
        str(args.calibration_limit),
        "--calibration-max-length",
        str(args.calibration_max_length),
        "--experiment-name-suffix",
        args.experiment_name_suffix,
    ]
    if args.save_dir:
        cmd.extend(["--save-dir", args.save_dir])

    print(f"[INFO] ROOT={root}")
    print(f"[INFO] MODEL_PATH={args.model_path}")
    print(f"[INFO] OUTPUT_DIR={args.output_dir}")
    print(f"[INFO] CONFIG_PATH={args.config_path}")
    print(f"[INFO] MIN_LAYER={args.min_layer} MAX_LAYER={args.max_layer} SPARSITY={args.sparsity}")
    print(
        f"[INFO] CALIBRATION_LIMIT={args.calibration_limit} "
        f"CALIBRATION_MAX_LENGTH={args.calibration_max_length}"
    )
    print(f"[INFO] EXPERIMENT_NAME_SUFFIX={args.experiment_name_suffix}")
    print(f"[INFO] PYTHON={sys.executable}")

    completed = subprocess.run(cmd, cwd=root)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
