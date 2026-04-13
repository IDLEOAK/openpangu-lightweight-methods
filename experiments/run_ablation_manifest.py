import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.run_ablation_variant_pipeline import REPO_ROOT, read_json, resolve_repo_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all variants from one ablation study manifest.")
    parser.add_argument("--manifest", required=True, help="Path to the study manifest.")
    parser.add_argument(
        "--results-root",
        default="",
        help="Override results root. Defaults to manifest.results_root_suggestion.",
    )
    parser.add_argument("--python-bin", default=sys.executable, help="Python interpreter used for subcommands.")
    parser.add_argument("--with-smoke", action="store_true", help="Run the smoke stage for variants that declare smoke_config.")
    parser.add_argument("--skip-completed", action="store_true", help="Skip variants/stages with existing outputs.")
    parser.add_argument(
        "--cleanup-heavy-artifacts",
        action="store_true",
        help="Delete exported_model and compressed_artifact for variants that finish the full pipeline.",
    )
    parser.add_argument(
        "--start-from",
        default="",
        help="Optional variant_id to start from. Earlier variants in the manifest are skipped.",
    )
    args = parser.parse_args()

    manifest_path = resolve_repo_path(args.manifest)
    manifest = read_json(manifest_path)

    started = not bool(args.start_from)
    for variant in manifest["variants"]:
        variant_id = variant["variant_id"]
        if not started:
            if variant_id == args.start_from:
                started = True
            else:
                print(f"[SKIP] before start-from: {variant_id}")
                continue

        cmd = [
            args.python_bin,
            str(REPO_ROOT / "experiments" / "run_ablation_variant_pipeline.py"),
            "--manifest",
            str(manifest_path),
            "--variant-id",
            variant_id,
        ]
        if args.results_root:
            cmd.extend(["--results-root", args.results_root])
        if args.with_smoke:
            cmd.append("--with-smoke")
        if args.skip_completed:
            cmd.append("--skip-completed")
        if args.cleanup_heavy_artifacts:
            cmd.append("--cleanup-heavy-artifacts")

        print(f"[RUN] variant={variant_id}")
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)

    print(f"[OK] manifest_complete={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
