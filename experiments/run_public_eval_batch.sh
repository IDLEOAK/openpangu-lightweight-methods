#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: bash experiments/run_public_eval_batch.sh <wikitext2|cnpublic> [python_bin]"
  exit 1
fi

DATASET_TAG="$1"
PYTHON_BIN="${2:-python}"

case "$DATASET_TAG" in
  wikitext2)
    CONFIGS=(
      "experiments/configs/sparsegpt_port_full34_wikitext2_eval.json"
      "experiments/configs/admm_port_full34_wikitext2_eval.json"
      "experiments/configs/gptq_port_full34_wikitext2_eval.json"
      "experiments/configs/awq_port_full34_wikitext2_eval.json"
      "experiments/configs/smoothquant_port_full34_wikitext2_eval.json"
    )
    ;;
  cnpublic)
    CONFIGS=(
      "experiments/configs/sparsegpt_port_full34_cnpublic_eval.json"
      "experiments/configs/admm_port_full34_cnpublic_eval.json"
      "experiments/configs/gptq_port_full34_cnpublic_eval.json"
      "experiments/configs/awq_port_full34_cnpublic_eval.json"
      "experiments/configs/smoothquant_port_full34_cnpublic_eval.json"
    )
    ;;
  *)
    echo "Unknown dataset tag: $DATASET_TAG"
    exit 1
    ;;
esac

for config in "${CONFIGS[@]}"; do
  echo "[RUN] $config"
  case "$config" in
    *sparsegpt*)
      "$PYTHON_BIN" experiments/sparsegpt/run_sparsegpt_scaffold.py --config "$config"
      ;;
    *admm*)
      "$PYTHON_BIN" experiments/admm/run_admm_scaffold.py --config "$config"
      ;;
    *gptq*)
      "$PYTHON_BIN" experiments/gptq/run_gptq_scaffold.py --config "$config"
      ;;
    *awq*)
      "$PYTHON_BIN" experiments/awq/run_awq_scaffold.py --config "$config"
      ;;
    *smoothquant*)
      "$PYTHON_BIN" experiments/smoothquant/run_smoothquant_scaffold.py --config "$config"
      ;;
  esac
done

echo "[DONE] dataset=$DATASET_TAG"
