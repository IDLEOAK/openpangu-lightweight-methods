# Export And Reload Validation Runbook

## Goal

This runbook covers the repository's export + reload roundtrip flow.
For external writing, thesis wording, and defense narration, use:

1. export a reloadable model
2. export the final compressed artifact
3. perform minimal reload verification

The internal flow is still split into two sub-steps:

1. reloadable checkpoint verification
2. compressed-artifact verification

## What Each Sub-Step Produces

1. `B1`
   - `exported_model/`
   - reloadable Hugging Face checkpoint
2. `B2`
   - `compressed_artifact/`
   - real compressed storage artifact
   - pruning methods: sparse mask + nonzero values
   - quantization methods: packed low-bit codes + scales / zeros (+ pre-scale when needed)

## Files Produced By This Flow

For each method run, this flow now writes:

1. `summary.json`
   - includes `saved_model_dir`
   - includes `saved_model_format`
   - includes `saved_model_info.total_size_bytes`
   - includes `compressed_artifact_dir`
   - includes `compressed_artifact_format`
   - includes `compressed_artifact_info.total_size_bytes`
2. `exported_model/`
   - the reloadable-checkpoint target directory
3. `compressed_artifact/`
   - the final compressed storage artifact
4. a separate reload verification run under `experiments/results/reload_verify/`
   - includes `reloaded_perplexity`
   - includes `reloaded_benchmark`
   - includes `source_comparison`
5. a separate compressed-artifact verification run under `experiments/results/compressed_verify/`
   - includes the same minimal `PPL + benchmark` verification on top of the base model

## Recommended Server Entry

Run from the repository root:

```bash
bash experiments/run_export_reload_roundtrip.sh sparsegpt /mnt/env/openpangu-publiceval-python.sh
bash experiments/run_export_reload_roundtrip.sh gptq /mnt/env/openpangu-publiceval-python.sh
bash experiments/run_export_reload_roundtrip.sh admm /mnt/env/openpangu-publiceval-python.sh
bash experiments/run_export_reload_roundtrip.sh awq /mnt/env/openpangu-publiceval-python.sh
bash experiments/run_export_reload_roundtrip.sh smoothquant /mnt/env/openpangu-publiceval-python.sh
```

Recommended order:

1. `sparsegpt`
2. `gptq`
3. `admm`
4. `awq`
5. `smoothquant`

## Minimal Verification Scope

The reload verification currently uses:

1. perplexity data: `experiments/data/formal_eval_prompts.jsonl` first 16 samples
2. benchmark data: `experiments/data/benchmarks/boolq_validation_mcq.jsonl` first 16 samples

This is only for artifact-loop verification, not for final thesis tables.
