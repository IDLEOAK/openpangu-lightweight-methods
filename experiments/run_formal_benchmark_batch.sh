#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: bash experiments/run_formal_benchmark_batch.sh <task-dir> [python_bin] [benchmark_limit]"
  exit 1
fi

TASK_DIR="$1"
PYTHON_BIN="${2:-python}"
BENCHMARK_LIMIT="${3:-0}"

BASELINE_CONFIG="experiments/configs/benchmark_public_mcq_baseline.json"

METHOD_RUNNERS=(
  "sparsegpt:experiments/configs/sparsegpt_port_full34_benchmark.json:experiments/sparsegpt/run_sparsegpt_scaffold.py"
  "admm:experiments/configs/admm_port_full34_benchmark.json:experiments/admm/run_admm_scaffold.py"
  "gptq:experiments/configs/gptq_port_full34_benchmark.json:experiments/gptq/run_gptq_scaffold.py"
  "awq:experiments/configs/awq_port_full34_benchmark.json:experiments/awq/run_awq_scaffold.py"
  "smoothquant:experiments/configs/smoothquant_port_full34_benchmark.json:experiments/smoothquant/run_smoothquant_scaffold.py"
)

for benchmark_data in "$TASK_DIR"/*.jsonl; do
  if [ ! -f "$benchmark_data" ]; then
    continue
  fi
  task_slug="$(basename "$benchmark_data" .jsonl)"
  echo "[RUN][baseline] $benchmark_data"
  if [ "$BENCHMARK_LIMIT" -gt 0 ]; then
    "$PYTHON_BIN" experiments/benchmark/run_benchmark_scaffold.py --config "$BASELINE_CONFIG" --benchmark-data "$benchmark_data" --limit "$BENCHMARK_LIMIT" --experiment-name-suffix "$task_slug"
  else
    "$PYTHON_BIN" experiments/benchmark/run_benchmark_scaffold.py --config "$BASELINE_CONFIG" --benchmark-data "$benchmark_data" --experiment-name-suffix "$task_slug"
  fi

  for entry in "${METHOD_RUNNERS[@]}"; do
    IFS=":" read -r method config runner <<< "$entry"
    echo "[RUN][$method] $benchmark_data"
    if [ "$BENCHMARK_LIMIT" -gt 0 ]; then
      "$PYTHON_BIN" "$runner" --config "$config" --benchmark-data "$benchmark_data" --benchmark-limit "$BENCHMARK_LIMIT" --experiment-name-suffix "$task_slug"
    else
      "$PYTHON_BIN" "$runner" --config "$config" --benchmark-data "$benchmark_data" --experiment-name-suffix "$task_slug"
    fi
  done
done

echo "[DONE] task_dir=$TASK_DIR"
