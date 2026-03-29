# OpenPangu Lightweight Experiments

This directory contains the shared scaffold for post-training pruning and quantization experiments on OpenPangu.

## Layout

- `configs/`: JSON experiment configs
- `data/`: small prompt/calibration samples
- `common/`: shared runtime, data, reporting, and metric helpers
- `sparsegpt/`: SparseGPT-oriented experiment entrypoints
- `gptq/`: GPTQ-oriented experiment entrypoints
- `results/`: generated run artifacts and summaries

## Current stage

The current scripts are scaffolds, not full algorithm ports.

They already provide:

1. unified config loading
2. local model/tokenizer loading
3. module inventory extraction for OpenPangu linear layers
4. baseline generation/perplexity measurement
5. run manifest and JSON result export

They do not yet apply SparseGPT or GPTQ updates to weights.

## Example

```powershell
C:\Tools\anaconda3\python.exe experiments\sparsegpt\run_sparsegpt_scaffold.py --config experiments\configs\sparsegpt_scaffold.json
C:\Tools\anaconda3\python.exe experiments\gptq\run_gptq_scaffold.py --config experiments\configs\gptq_scaffold.json
```
