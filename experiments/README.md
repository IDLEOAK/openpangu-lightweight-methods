# OpenPangu Lightweight Experiments

This directory contains the shared scaffold for post-training pruning and quantization experiments on OpenPangu, plus the current final-artifact benchmark and ablation orchestration pipeline.

## Layout

- `configs/`: JSON experiment configs
- `configs/ablation/`: ablation manifests and per-variant configs
- `data/`: prompt, calibration, public-eval, and benchmark samples
- `common/`: shared runtime, data, reporting, and metric helpers
- `benchmark/`: public benchmark evaluation entrypoints
- `export_reload/`: exported-model and compressed-artifact reload verification
- `compressed_artifacts/`: artifact serialization helpers
- `sparsegpt/`: SparseGPT-oriented experiment entrypoints
- `admm/`: ADMM pruning-oriented experiment entrypoints
- `llm_bip/`: LLM-BIP structured pruning-oriented experiment entrypoints
- `gptq/`: GPTQ-oriented experiment entrypoints
- `awq/`: AWQ-oriented experiment entrypoints
- `smoothquant/`: SmoothQuant-oriented experiment entrypoints
- `run_final_artifact_benchmark_batch.py`: final hard-8x8 artifact benchmark batch runner
- `build_final_artifact_benchmark_summary.py`: final artifact benchmark aggregator
- `run_ablation_variant_pipeline.py`: one-variant ablation pipeline
- `run_ablation_manifest.py`: study-level ablation batch runner
- `build_ablation_result_summary.py`: ablation summary aggregator
- `cleanup_ablation_variant_artifacts.py`: optional cleanup for completed ablation variants
- `results/`: generated run artifacts and summaries

## Current status

The experiment framework is no longer only a scaffold. It now provides:

1. unified config loading
2. local model/tokenizer loading
3. module inventory extraction for OpenPangu linear layers
4. baseline generation and perplexity measurement
5. SparseGPT sequential pruning for OpenPangu
6. ADMM sequential pruning for OpenPangu
7. LLM-BIP structured pruning proxy for OpenPangu
8. GPTQ sequential quantization for OpenPangu
9. AWQ sequential quantization for OpenPangu
10. SmoothQuant-inspired sequential quantization for OpenPangu
11. exported-model and compressed-artifact export
12. minimal reload verification for exported models and compressed artifacts
13. public multiple-choice benchmark scaffold for baseline and reloadable model directories
14. final `MMLU hard 8 + C-Eval hard 8` artifact benchmark batch and summary generation
15. manifest-driven ablation execution, artifact benchmark, and study-level result aggregation

## Result entrypoints

1. `results/current_result_summary.{json,md}` summarizes the prompt-eval / generation snapshot across the tracked main methods.
2. `results/benchmark_result_summary.{json,md}` summarizes the legacy in-memory 15-task benchmark.
3. `results/final_artifact_benchmark_summary.{json,md}` is the current final-artifact hard-8x8 benchmark entrypoint.
4. `results/ablation/*/*_summary.{json,md}` stores the study-level ablation summaries.

## Example commands

Build local summaries:

```powershell
C:\Tools\anaconda3\python.exe experiments\build_current_result_summary.py
C:\Tools\anaconda3\python.exe experiments\build_benchmark_result_summary.py
C:\Tools\anaconda3\python.exe experiments\build_final_artifact_benchmark_summary.py
```

Run a full34 method config locally:

```powershell
C:\Tools\anaconda3\python.exe experiments\sparsegpt\run_sparsegpt_scaffold.py --config experiments\configs\sparsegpt_port_full34_formal_local.json
C:\Tools\anaconda3\python.exe experiments\gptq\run_gptq_scaffold.py --config experiments\configs\gptq_port_full34_formal_local.json
```

Run the final artifact hard-8x8 benchmark on the server:

```bash
/mnt/env/openpangu-publiceval-python.sh experiments/run_final_artifact_benchmark_batch.py /mnt/env/openpangu-publiceval-python.sh
```

Run one ablation study manifest:

```powershell
C:\Tools\anaconda3\python.exe experiments\run_ablation_manifest.py --manifest experiments\configs\ablation\pruning_sparsity_manifest.json
```

## Related runbooks

1. `EXPORT_RELOAD_ROUNDTRIP_RUNBOOK.md`
2. `FINAL_ARTIFACT_BENCHMARK_RUNBOOK.md`
