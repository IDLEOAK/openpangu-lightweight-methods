#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: bash experiments/run_benchmark_batch.sh <task-set-dir> [python_bin]"
  exit 1
fi

TASK_DIR="$1"
PYTHON_BIN="${2:-python}"
CONFIG="experiments/configs/benchmark_public_mcq_baseline.json"

for benchmark_data in "$TASK_DIR"/*.jsonl; do
  if [ ! -f "$benchmark_data" ]; then
    continue
  fi
  echo "[RUN] $benchmark_data"
  "$PYTHON_BIN" experiments/benchmark/run_benchmark_scaffold.py --config "$CONFIG" --benchmark-data "$benchmark_data"
done

echo "[DONE] task_dir=$TASK_DIR"
