# Final Artifact Benchmark Runbook

## Goal

This runbook covers the repository's internal `stage_c` benchmark flow.
For external writing, thesis wording, and defense narration, it is better described as:

1. benchmark of the final compressed artifacts
2. balanced `MMLU 4 + C-Eval 4` task verification

The current flow evaluates the final compressed artifacts on the narrowed benchmark line:

1. `MMLU` 4-task English subset
2. `C-Eval` 4-task Chinese subset

The baseline uses the original dense model.
All compressed methods use the `compressed_artifact/` outputs produced by the internal export/reload flow.

## Task Set

The current internal task group is `stage_c_core`, containing 8 tasks in a balanced `4 + 4` layout:

1. `mmlu_college_computer_science`
2. `mmlu_computer_security`
3. `mmlu_machine_learning`
4. `mmlu_formal_logic`
5. `ceval_college_programming`
6. `ceval_computer_network`
7. `ceval_operating_system`
8. `ceval_computer_architecture`

## Execution

Run from the repository root:

```bash
/mnt/env/openpangu-publiceval-python.sh experiments/run_stage_c_benchmark_batch.py /mnt/env/openpangu-publiceval-python.sh
```

## Output Layout

Intermediate run dirs are written to:

```text
/root/stage_c_work/stage_c_results/
```

Final per-task run dirs are copied to:

```text
/mnt/openpangu-embedded-7b-model/experiments/results/stage_c_benchmark/
```

Final summary files are written to:

```text
/mnt/results/stage_c/stage_c_benchmark_summary.json
/mnt/results/stage_c/stage_c_benchmark_summary.md
```

## Notes

1. This final-artifact benchmark line is intentionally narrower than the earlier 15-task transition benchmark, but more balanced than the old `6 + 2` split.
2. It evaluates the final compressed artifacts, not the old in-memory-only method states.
3. If a method needs to be rerun, update `experiments/configs/stage_c_artifact_manifest.json` to point at the latest `compressed_artifact/` directory.
4. If the new `C-Eval` task jsonl files do not yet exist locally, rerun:

```bash
/mnt/env/openpangu-publiceval-python.sh experiments/data/prepare_benchmark_sets.py --tasks stage_c_core --max-samples-per-task 0
```
