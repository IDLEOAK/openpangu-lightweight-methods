import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional


REPO_ROOT = Path(__file__).resolve().parent.parent

RUNNER_BY_METHOD = {
    "sparsegpt": REPO_ROOT / "experiments" / "sparsegpt" / "run_sparsegpt_scaffold.py",
    "admm": REPO_ROOT / "experiments" / "admm" / "run_admm_scaffold.py",
    "gptq": REPO_ROOT / "experiments" / "gptq" / "run_gptq_scaffold.py",
    "awq": REPO_ROOT / "experiments" / "awq" / "run_awq_scaffold.py",
    "smoothquant": REPO_ROOT / "experiments" / "smoothquant" / "run_smoothquant_scaffold.py",
}


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def load_variant(manifest_path: Path, variant_id: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
    manifest = read_json(manifest_path)
    for entry in manifest["variants"]:
        if entry["variant_id"] == variant_id:
            return manifest, entry
    raise ValueError(f"variant_id not found in manifest: {variant_id}")


def latest_run_summary(stage_root: Path, method_dir: str) -> Optional[Path]:
    method_root = stage_root / method_dir
    if not method_root.exists():
        return None
    candidates = sorted(method_root.glob("*/summary.json"), key=lambda item: item.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def stage_complete(summary_path: Optional[Path]) -> bool:
    return summary_path is not None and summary_path.exists()


def merged_stage_overrides(variant: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
    method_overrides = dict(variant.get("method_overrides", {}))
    calibration_overrides = dict(variant.get("calibration_overrides", {}))
    module_selection_overrides = dict(variant.get("module_selection_overrides", {}))
    if extra:
        method_overrides.update(extra.get("method_overrides", {}))
        calibration_overrides.update(extra.get("calibration_overrides", {}))
        module_selection_overrides.update(extra.get("module_selection_overrides", {}))
    return {
        "method_overrides": method_overrides,
        "calibration_overrides": calibration_overrides,
        "module_selection_overrides": module_selection_overrides,
    }


def materialize_stage_config(
    base_config: Path,
    method: str,
    variant: Dict[str, Any],
    generated_config_dir: Path,
    stage_label: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    payload = read_json(base_config)
    overrides = merged_stage_overrides(variant, extra)

    method_plan = dict(payload.get(method, {}))
    method_plan.update(overrides["method_overrides"])
    payload[method] = method_plan

    calibration_cfg = dict(payload.get("calibration_data", {}))
    calibration_cfg.update(overrides["calibration_overrides"])
    payload["calibration_data"] = calibration_cfg

    module_selection = dict(payload.get("module_selection", {}))
    module_selection.update(overrides["module_selection_overrides"])
    payload["module_selection"] = module_selection

    generated_config_dir.mkdir(parents=True, exist_ok=True)
    generated_config = generated_config_dir / f"{stage_label}_{variant['variant_id']}.json"
    write_json(generated_config, payload)
    return generated_config


def run_command(cmd: list[str]) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def ensure_cleanup_target(path: Path, variant_root: Path) -> Path:
    target = path.resolve()
    variant_root_resolved = variant_root.resolve()
    try:
        target.relative_to(variant_root_resolved)
    except ValueError as exc:
        raise RuntimeError(f"cleanup target escapes variant root: {target}") from exc
    return target


def cleanup_variant_artifacts(variant_root: Path, export_summary: Dict[str, Any]) -> None:
    cleanup_targets = []
    for field in ("saved_model_dir", "compressed_artifact_dir"):
        value = export_summary.get(field)
        if not value:
            continue
        cleanup_targets.append(ensure_cleanup_target(Path(value), variant_root))

    if not cleanup_targets:
        print(f"[SKIP] cleanup_no_targets variant_root={variant_root}")
        return

    for target in cleanup_targets:
        if not target.exists():
            print(f"[SKIP] cleanup_missing={target}")
            continue
        shutil.rmtree(target)
        print(f"[OK] cleanup_removed={target}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the full ablation pipeline for one variant.")
    parser.add_argument("--manifest", required=True, help="Path to the study manifest.")
    parser.add_argument("--variant-id", required=True, help="Variant id in the manifest.")
    parser.add_argument(
        "--results-root",
        default="",
        help="Override results root. Defaults to manifest.results_root_suggestion.",
    )
    parser.add_argument("--python-bin", default=sys.executable, help="Python interpreter used for subcommands.")
    parser.add_argument("--with-smoke", action="store_true", help="Run the smoke stage if smoke_config is declared.")
    parser.add_argument("--skip-completed", action="store_true", help="Skip stages with an existing latest summary.")
    parser.add_argument(
        "--cleanup-heavy-artifacts",
        action="store_true",
        help="Delete exported_model and compressed_artifact after the full variant pipeline succeeds.",
    )
    args = parser.parse_args()

    manifest_path = resolve_repo_path(args.manifest)
    manifest, variant = load_variant(manifest_path, args.variant_id)
    results_root_value = args.results_root or manifest.get("results_root_suggestion", "")
    if not results_root_value:
        raise ValueError("results root is required either via --results-root or manifest.results_root_suggestion")
    results_root = Path(results_root_value).resolve()

    variant_root = results_root / variant["result_subdir"]
    variant_root.mkdir(parents=True, exist_ok=True)
    method = variant["method"]
    runner = RUNNER_BY_METHOD[method]
    generated_config_dir = variant_root / "_generated_configs"

    smoke_config_value = variant.get("smoke_config")
    if args.with_smoke and smoke_config_value:
        smoke_config = materialize_stage_config(
            resolve_repo_path(smoke_config_value),
            method,
            variant,
            generated_config_dir,
            "smoke",
            variant.get("smoke_overrides", {}),
        )
        smoke_stage_root = variant_root / "smoke"
        smoke_summary = latest_run_summary(smoke_stage_root, method)
        if not (args.skip_completed and stage_complete(smoke_summary)):
            cmd = [
                args.python_bin,
                str(runner),
                "--config",
                str(smoke_config),
                "--output-dir",
                str(smoke_stage_root),
                "--experiment-name-suffix",
                variant["variant_id"],
            ]
            run_command(cmd)

    formal_config = materialize_stage_config(
        resolve_repo_path(variant["formal_config"]),
        method,
        variant,
        generated_config_dir,
        "formal",
    )
    formal_stage_root = variant_root / "export_bundle"
    formal_summary = latest_run_summary(formal_stage_root, method)
    if not (args.skip_completed and stage_complete(formal_summary)):
        cmd = [
            args.python_bin,
            str(runner),
            "--config",
            str(formal_config),
            "--output-dir",
            str(formal_stage_root),
            "--experiment-name-suffix",
            variant["variant_id"],
        ]
        run_command(cmd)
        formal_summary = latest_run_summary(formal_stage_root, method)

    if formal_summary is None:
        raise RuntimeError(f"export bundle summary not found for variant={variant['variant_id']}")
    export_summary = read_json(formal_summary)
    export_run_dir = formal_summary.parent
    exported_model_dir = export_summary.get("saved_model_dir")
    compressed_artifact_dir = export_summary.get("compressed_artifact_dir")
    if not exported_model_dir:
        raise RuntimeError(f"saved_model_dir missing in {formal_summary}")
    if not compressed_artifact_dir:
        raise RuntimeError(f"compressed_artifact_dir missing in {formal_summary}")

    wikitext2_config = materialize_stage_config(
        resolve_repo_path(variant["wikitext2_config"]),
        method,
        variant,
        generated_config_dir,
        "wikitext2",
    )
    wikitext2_stage_root = variant_root / "wikitext2_eval"
    wikitext2_summary = latest_run_summary(wikitext2_stage_root, method)
    if not (args.skip_completed and stage_complete(wikitext2_summary)):
        cmd = [
            args.python_bin,
            str(runner),
            "--config",
            str(wikitext2_config),
            "--output-dir",
            str(wikitext2_stage_root),
            "--experiment-name-suffix",
            variant["variant_id"],
        ]
        run_command(cmd)

    cnpublic_config = materialize_stage_config(
        resolve_repo_path(variant["cnpublic_config"]),
        method,
        variant,
        generated_config_dir,
        "cnpublic",
    )
    cnpublic_stage_root = variant_root / "cnpublic_eval"
    cnpublic_summary = latest_run_summary(cnpublic_stage_root, method)
    if not (args.skip_completed and stage_complete(cnpublic_summary)):
        cmd = [
            args.python_bin,
            str(runner),
            "--config",
            str(cnpublic_config),
            "--output-dir",
            str(cnpublic_stage_root),
            "--experiment-name-suffix",
            variant["variant_id"],
        ]
        run_command(cmd)

    reload_stage_root = variant_root / "reload_verify"
    reload_summary = latest_run_summary(reload_stage_root, "reload_verify")
    if not (args.skip_completed and stage_complete(reload_summary)):
        cmd = [
            args.python_bin,
            str(REPO_ROOT / "experiments" / "export_reload" / "run_reload_verification.py"),
            "--config",
            str(REPO_ROOT / "experiments" / "configs" / "reload_verification_minimal.json"),
            "--model-path",
            str(exported_model_dir),
            "--output-dir",
            str(reload_stage_root),
            "--source-summary",
            str(formal_summary),
            "--experiment-name-suffix",
            variant["variant_id"],
        ]
        run_command(cmd)

    compressed_stage_root = variant_root / "compressed_verify"
    compressed_summary = latest_run_summary(compressed_stage_root, "compressed_verify")
    if not (args.skip_completed and stage_complete(compressed_summary)):
        cmd = [
            args.python_bin,
            str(REPO_ROOT / "experiments" / "export_reload" / "run_compressed_artifact_verification.py"),
            "--config",
            str(REPO_ROOT / "experiments" / "configs" / "reload_verification_minimal.json"),
            "--base-model-path",
            str(REPO_ROOT),
            "--artifact-dir",
            str(compressed_artifact_dir),
            "--output-dir",
            str(compressed_stage_root),
            "--source-summary",
            str(formal_summary),
            "--experiment-name-suffix",
            variant["variant_id"],
        ]
        run_command(cmd)

    artifact_summary_path = variant_root / "artifact_benchmark" / "summary" / "artifact_benchmark_summary.json"
    if not (args.skip_completed and artifact_summary_path.exists()):
        cmd = [
            args.python_bin,
            str(REPO_ROOT / "experiments" / "run_ablation_artifact_benchmark_batch.py"),
            "--python-bin",
            args.python_bin,
            "--method-label",
            variant.get("artifact_method_label", variant["variant_id"]),
            "--artifact-dir",
            str(compressed_artifact_dir),
            "--result-root",
            str(variant_root),
        ]
        run_command(cmd)

    cmd = [
        args.python_bin,
        str(REPO_ROOT / "experiments" / "build_ablation_result_summary.py"),
        "--manifest",
        str(manifest_path),
        "--results-root",
        str(results_root),
        "--output-dir",
        str(results_root),
    ]
    run_command(cmd)

    if args.cleanup_heavy_artifacts:
        cleanup_variant_artifacts(variant_root, export_summary)

    print(f"[OK] variant_complete={variant['variant_id']}")
    print(f"[OK] export_run_dir={export_run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
