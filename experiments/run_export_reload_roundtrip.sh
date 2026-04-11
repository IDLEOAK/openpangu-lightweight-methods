#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: bash experiments/run_export_reload_roundtrip.sh <method> [python_bin]"
  echo "Methods: sparsegpt gptq admm awq smoothquant"
  exit 1
fi

METHOD="$1"
PYTHON_BIN="${2:-python}"
KEEP_DENSE_EXPORT="${KEEP_DENSE_EXPORT:-0}"

case "$METHOD" in
  sparsegpt)
    EXPORT_CONFIG="experiments/configs/sparsegpt_port_full34_export_bundle.json"
    RUNNER="experiments/sparsegpt/run_sparsegpt_scaffold.py"
    RESULT_DIR="experiments/results/sparsegpt"
    ;;
  gptq)
    EXPORT_CONFIG="experiments/configs/gptq_port_full34_export_bundle.json"
    RUNNER="experiments/gptq/run_gptq_scaffold.py"
    RESULT_DIR="experiments/results/gptq"
    ;;
  admm)
    EXPORT_CONFIG="experiments/configs/admm_port_full34_export_bundle.json"
    RUNNER="experiments/admm/run_admm_scaffold.py"
    RESULT_DIR="experiments/results/admm"
    ;;
  awq)
    EXPORT_CONFIG="experiments/configs/awq_port_full34_export_bundle.json"
    RUNNER="experiments/awq/run_awq_scaffold.py"
    RESULT_DIR="experiments/results/awq"
    ;;
  smoothquant)
    EXPORT_CONFIG="experiments/configs/smoothquant_port_full34_export_bundle.json"
    RUNNER="experiments/smoothquant/run_smoothquant_scaffold.py"
    RESULT_DIR="experiments/results/smoothquant"
    ;;
  *)
    echo "[ERROR] Unsupported method: $METHOD"
    exit 1
    ;;
esac

EXPERIMENT_NAME="$("$PYTHON_BIN" - <<'PY' "$EXPORT_CONFIG"
import json
import sys
from pathlib import Path
config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(config["experiment_name"])
PY
)"

echo "[RUN][$METHOD] export"
"$PYTHON_BIN" "$RUNNER" --config "$EXPORT_CONFIG"

RUN_DIR="$(ls -dt "$RESULT_DIR"/*"$EXPERIMENT_NAME"* 2>/dev/null | head -n 1)"
if [ -z "$RUN_DIR" ]; then
  echo "[ERROR] Failed to locate run_dir for $METHOD under $RESULT_DIR"
  exit 1
fi

EXPORTED_MODEL_DIR="$RUN_DIR/exported_model"
if [ ! -d "$EXPORTED_MODEL_DIR" ]; then
  echo "[ERROR] Exported model directory not found: $EXPORTED_MODEL_DIR"
  exit 1
fi

COMPRESSED_ARTIFACT_DIR="$RUN_DIR/compressed_artifact"
if [ ! -d "$COMPRESSED_ARTIFACT_DIR" ]; then
  echo "[ERROR] Compressed artifact directory not found: $COMPRESSED_ARTIFACT_DIR"
  exit 1
fi

echo "[RUN][$METHOD] b1_reload_verify"
"$PYTHON_BIN" experiments/export_reload/run_reload_verification.py \
  --config experiments/configs/reload_verification_minimal.json \
  --model-path "$EXPORTED_MODEL_DIR" \
  --source-summary "$RUN_DIR/summary.json" \
  --experiment-name-suffix "${METHOD}_$(basename "$RUN_DIR")"

echo "[RUN][$METHOD] b2_compressed_verify"
"$PYTHON_BIN" experiments/export_reload/run_compressed_artifact_verification.py \
  --config experiments/configs/reload_verification_minimal.json \
  --base-model-path . \
  --artifact-dir "$COMPRESSED_ARTIFACT_DIR" \
  --source-summary "$RUN_DIR/summary.json" \
  --experiment-name-suffix "${METHOD}_compressed_$(basename "$RUN_DIR")"

if [ "$KEEP_DENSE_EXPORT" != "1" ]; then
  rm -rf "$EXPORTED_MODEL_DIR"
  echo "[DONE][$METHOD] removed_dense_export=$EXPORTED_MODEL_DIR"
fi

echo "[DONE][$METHOD] run_dir=$RUN_DIR"
echo "[DONE][$METHOD] exported_model_dir=$EXPORTED_MODEL_DIR"
echo "[DONE][$METHOD] compressed_artifact_dir=$COMPRESSED_ARTIFACT_DIR"
