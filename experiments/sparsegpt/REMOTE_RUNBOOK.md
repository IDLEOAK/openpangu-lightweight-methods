# SparseGPT Remote Runbook

## Purpose

This runbook records the early remote SparseGPT minimal-run path after copying a prepared bundle to a Linux server. It remains useful for backtracking environment setup or revalidating a new remote machine, but the maintained experiment line has moved to full-34-layer export, public-corpus PPL evaluation, reload verification, and final artifact benchmarking.

## Expected Remote Layout

Place the bundle contents on the server, then overlay the included `openpangu-embedded-7b-model/` files onto the remote model directory that already contains the OpenPangu weights.

The remote model directory should look like:

```text
<REMOTE_MODEL_DIR>/
  model-00001-of-00009.safetensors
  ...
  modeling_openpangu_dense.py
  modular_openpangu_dense.py
  experiments/
```

## One-Stop Python Entry

Use `experiments/sparsegpt/remote_sparsegpt_job.py` as the only remote entry.

This file already includes a configuration block at the top. In the default form:

1. `MODEL_PATH` is fixed to `/home/think4090/jlj/openpangu-embedded-7b-model`.
2. `HF_HOME` is fixed to `/home/think4090/jlj/openpangu-embedded-7b-model/.hf_cache`.
3. `OUTPUT_DIR` is fixed to `/home/think4090/jlj/openpangu-embedded-7b-model/experiments/results`.
4. `MIN_LAYER=0` and `MAX_LAYER=1`, which means the minimal verified run only touches layer 0.
5. `ARCHIVE_RESULTS=True`, so the result directory is automatically packaged after the run succeeds.
6. `ARCHIVE_PATH` is fixed to `/home/think4090/jlj/openpangu-embedded-7b-model/experiments/results/sparsegpt/server_minimal_result.tar.gz`.

If the remote directory layout matches the expected structure, you should be able to run the file directly without editing the path fields.

## Current Workflow Note

For current full-34-layer experiments, prefer the shared config and runbook entrypoints under `experiments/`:

1. `experiments/configs/sparsegpt_port_full34_export_bundle.json` for full-model export artifact generation
2. `experiments/EXPORT_RELOAD_ROUNDTRIP_RUNBOOK.md` for exported-model and compressed-artifact reload checks
3. `experiments/FINAL_ARTIFACT_BENCHMARK_RUNBOOK.md` for the final `MMLU hard 8 + C-Eval hard 8` artifact benchmark
4. `experiments/results/final_artifact_benchmark_summary.{json,md}` for the maintained benchmark result entrypoint

The gradual layer expansion below is only needed when validating a fresh remote environment from the minimal script:

1. first `MIN_LAYER=0`, `MAX_LAYER=4`
2. then continue to larger ranges only if the previous run is stable
3. keep `SPARSITY`, sample count and token length fixed while checking expansion stability

## Result Packaging

The one-stop script already packages the newest completed run directory automatically when `ARCHIVE_RESULTS=True`.

After a successful run, the files that matter are:

1. the generated result directory under `experiments/results/sparsegpt/`
2. the generated `.tar.gz` archive
3. `remote_job_manifest.json` inside the run directory

The file that should be sent back is the generated `.tar.gz`.
