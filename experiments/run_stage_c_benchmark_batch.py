import json
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def latest_run(output_root: Path, method: str, experiment_name: str, task_slug: str) -> Path:
    pattern = f"*{experiment_name}_{task_slug}"
    matches = sorted((output_root / method).glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise RuntimeError(f"Run dir not found for method={method} task={task_slug}")
    return matches[0]


def main() -> int:
    python_bin = sys.argv[1] if len(sys.argv) > 1 else sys.executable

    task_manifest = read_json(REPO_ROOT / "experiments" / "configs" / "stage_c_task_manifest.json")
    method_manifest = read_json(REPO_ROOT / "experiments" / "configs" / "stage_c_artifact_manifest.json")
    benchmark_config = REPO_ROOT / "experiments" / "configs" / "stage_c_benchmark_base.json"

    process_root = Path(method_manifest["process_results_root"])
    final_results_root = (REPO_ROOT / method_manifest["final_results_root"]).resolve()
    final_summary_root = Path(method_manifest["final_summary_root"])
    process_root.mkdir(parents=True, exist_ok=True)
    final_results_root.mkdir(parents=True, exist_ok=True)
    final_summary_root.mkdir(parents=True, exist_ok=True)

    experiment_name = read_json(benchmark_config)["experiment_name"]
    methods = {"baseline": None}
    methods.update({name: info["artifact_dir"] for name, info in method_manifest["methods"].items()})

    runner = REPO_ROOT / "experiments" / "benchmark" / "run_stage_c_model_benchmark.py"
    for task in task_manifest["tasks"]:
        benchmark_data = task["benchmark_data"]
        task_slug = Path(benchmark_data).stem
        for method, artifact_dir in methods.items():
            cmd = [
                python_bin,
                str(runner),
                "--config",
                str(benchmark_config),
                "--benchmark-data",
                benchmark_data,
                "--output-dir",
                str(process_root),
                "--method-label",
                method,
            ]
            if artifact_dir is not None:
                cmd.extend(["--artifact-dir", artifact_dir])
            print("[RUN]", method, task_slug)
            subprocess.run(cmd, cwd=REPO_ROOT, check=True)
            src_run = latest_run(process_root, method, experiment_name, task_slug)
            dst_method_root = final_results_root / method
            dst_method_root.mkdir(parents=True, exist_ok=True)
            dst_run = dst_method_root / src_run.name
            if dst_run.exists():
                shutil.rmtree(dst_run)
            shutil.copytree(src_run, dst_run)

    summary_builder = REPO_ROOT / "experiments" / "build_stage_c_benchmark_summary.py"
    subprocess.run([python_bin, str(summary_builder)], cwd=REPO_ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
