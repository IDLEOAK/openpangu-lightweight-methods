import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.build_ablation_result_summary import collect_variant_row
from experiments.run_ablation_variant_pipeline import cleanup_variant_artifacts, read_json, resolve_repo_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Delete heavy exported artifacts for ablation variants that already completed the full pipeline."
    )
    parser.add_argument("--manifest", required=True, help="Path to the ablation study manifest.")
    parser.add_argument(
        "--results-root",
        default="",
        help="Override for the ablation results root. Defaults to manifest.results_root_suggestion.",
    )
    parser.add_argument(
        "--variant-id",
        action="append",
        default=[],
        help="Optional variant_id filter. Repeat to clean multiple explicit variants.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete the heavy artifacts. Without this flag the script only reports candidates.",
    )
    args = parser.parse_args()

    manifest_path = resolve_repo_path(args.manifest)
    manifest = read_json(manifest_path)
    results_root_value = args.results_root or manifest.get("results_root_suggestion", "")
    if not results_root_value:
        raise ValueError("results root is required either via --results-root or manifest.results_root_suggestion")
    results_root = Path(results_root_value).resolve()
    selected_variant_ids = set(args.variant_id)

    for entry in manifest["variants"]:
        variant_id = entry["variant_id"]
        if selected_variant_ids and variant_id not in selected_variant_ids:
            continue

        row = collect_variant_row(entry, manifest, results_root)
        if row["missing_stages"]:
            print(f"[SKIP] variant_incomplete={variant_id} missing={','.join(row['missing_stages'])}")
            continue

        export_summary_path_value = row["stage_paths"]["export_bundle_summary"]
        if not export_summary_path_value:
            print(f"[SKIP] export_summary_missing={variant_id}")
            continue

        export_summary = read_json(Path(export_summary_path_value))
        for field in ("saved_model_dir", "compressed_artifact_dir"):
            value = export_summary.get(field)
            print(f"[INFO] {variant_id} {field}={value or '--'}")

        if not args.execute:
            continue

        cleanup_variant_artifacts(Path(row["result_root"]), export_summary)
        print(f"[OK] cleanup_complete={variant_id}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
