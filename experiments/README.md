# OpenPangu Lightweight Experiments

This directory contains the shared scaffold for post-training pruning and quantization experiments on OpenPangu.

## Layout

- `configs/`: JSON experiment configs
- `data/`: small prompt/calibration samples
- `common/`: shared runtime, data, reporting, and metric helpers
- `sparsegpt/`: SparseGPT-oriented experiment entrypoints
- `admm/`: ADMM pruning-oriented experiment entrypoints
- `llm_bip/`: LLM-BIP structured pruning-oriented experiment entrypoints
- `gptq/`: GPTQ-oriented experiment entrypoints
- `awq/`: AWQ-oriented experiment entrypoints
- `smoothquant/`: SmoothQuant-oriented experiment entrypoints
- `results/`: generated run artifacts and summaries

## Current stage

The experiment framework is no longer only a scaffold.

It now provides:

1. unified config loading
2. local model/tokenizer loading
3. module inventory extraction for OpenPangu linear layers
4. baseline generation/perplexity measurement
5. SparseGPT sequential pruning for OpenPangu
6. ADMM sequential pruning for OpenPangu
7. LLM-BIP structured pruning proxy for OpenPangu
8. GPTQ sequential quantization for OpenPangu
9. AWQ sequential quantization for OpenPangu
10. SmoothQuant-inspired sequential quantization for OpenPangu
11. run manifest and JSON result export

Current project-level conclusions:

1. SparseGPT has completed minimal, multi-layer, and full34 execution.
2. GPTQ has completed minimal, multi-layer, and full34 execution.
3. `results/current_result_summary.{json,md}` is the canonical summary entry for the latest unified snapshot.

## Example

```powershell
C:\Tools\anaconda3\python.exe experiments\sparsegpt\run_sparsegpt_scaffold.py --config experiments\configs\sparsegpt_scaffold.json
C:\Tools\anaconda3\python.exe experiments\gptq\run_gptq_scaffold.py --config experiments\configs\gptq_scaffold.json
```

For the currently meaningful full-model runs, use:

```powershell
C:\Tools\anaconda3\python.exe experiments\sparsegpt\run_sparsegpt_scaffold.py --config experiments\configs\sparsegpt_port_full34_formal_local.json
C:\Tools\anaconda3\python.exe experiments\gptq\run_gptq_scaffold.py --config experiments\configs\gptq_port_full34_formal_local.json
```

For newly added local horizontal-comparison entrypoints, use:

```powershell
C:\Tools\anaconda3\python.exe experiments\admm\run_admm_scaffold.py --config experiments\configs\admm_port_minimal.json
C:\Tools\anaconda3\python.exe experiments\awq\run_awq_scaffold.py --config experiments\configs\awq_port_minimal.json
C:\Tools\anaconda3\python.exe experiments\llm_bip\run_llm_bip_scaffold.py --config experiments\configs\llm_bip_port_minimal.json
C:\Tools\anaconda3\python.exe experiments\smoothquant\run_smoothquant_scaffold.py --config experiments\configs\smoothquant_port_minimal.json
```
