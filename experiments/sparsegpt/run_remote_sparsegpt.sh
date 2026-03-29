#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-$ROOT}"
HF_HOME="${HF_HOME:-$ROOT/.hf_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/experiments/results}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT/experiments/configs/sparsegpt_port_minimal.json}"
MIN_LAYER="${MIN_LAYER:-0}"
MAX_LAYER="${MAX_LAYER:-1}"
SPARSITY="${SPARSITY:-0.3}"
CALIBRATION_LIMIT="${CALIBRATION_LIMIT:-2}"
CALIBRATION_MAX_LENGTH="${CALIBRATION_MAX_LENGTH:-128}"
EXPERIMENT_NAME_SUFFIX="${EXPERIMENT_NAME_SUFFIX:-remote}"
SAVE_DIR="${SAVE_DIR:-}"

mkdir -p "$HF_HOME"
mkdir -p "$OUTPUT_DIR"

CMD=(
  "$PYTHON_BIN"
  "$ROOT/experiments/sparsegpt/run_sparsegpt_scaffold.py"
  --config "$CONFIG_PATH"
  --model-path "$MODEL_PATH"
  --hf-home "$HF_HOME"
  --output-dir "$OUTPUT_DIR"
  --min-layer "$MIN_LAYER"
  --max-layer "$MAX_LAYER"
  --sparsity "$SPARSITY"
  --calibration-limit "$CALIBRATION_LIMIT"
  --calibration-max-length "$CALIBRATION_MAX_LENGTH"
  --experiment-name-suffix "$EXPERIMENT_NAME_SUFFIX"
)

if [[ -n "$SAVE_DIR" ]]; then
  CMD+=(--save-dir "$SAVE_DIR")
fi

printf '[INFO] ROOT=%s\n' "$ROOT"
printf '[INFO] MODEL_PATH=%s\n' "$MODEL_PATH"
printf '[INFO] OUTPUT_DIR=%s\n' "$OUTPUT_DIR"
printf '[INFO] CONFIG_PATH=%s\n' "$CONFIG_PATH"
printf '[INFO] MIN_LAYER=%s MAX_LAYER=%s SPARSITY=%s\n' "$MIN_LAYER" "$MAX_LAYER" "$SPARSITY"
printf '[INFO] CALIBRATION_LIMIT=%s CALIBRATION_MAX_LENGTH=%s\n' "$CALIBRATION_LIMIT" "$CALIBRATION_MAX_LENGTH"
printf '[INFO] EXPERIMENT_NAME_SUFFIX=%s\n' "$EXPERIMENT_NAME_SUFFIX"

"${CMD[@]}"
