# Final Artifact Benchmark Runbook

## Goal

This runbook covers benchmark verification for the final compressed artifacts.

The current maintained task line is:

1. `MMLU hard 8` English subjects
2. `C-Eval hard 8` Chinese subjects
3. official-reference few-shot prompt format for both families
4. no outer chat-template wrapping for exported MMLU / C-Eval benchmark prompts

The baseline uses the original dense model.
Compressed methods use the `compressed_artifact/` outputs produced by the export + reload roundtrip flow.

## Task Set

The current task manifest is `experiments/configs/final_artifact_benchmark_tasks.json`.

It points to 16 tasks:

1. `mmlu_abstract_algebra`
2. `mmlu_college_mathematics`
3. `mmlu_high_school_statistics`
4. `mmlu_college_chemistry`
5. `mmlu_college_physics`
6. `mmlu_high_school_mathematics`
7. `mmlu_high_school_chemistry`
8. `mmlu_high_school_physics`
9. `ceval_advanced_mathematics`
10. `ceval_discrete_mathematics`
11. `ceval_probability_and_statistics`
12. `ceval_college_chemistry`
13. `ceval_college_physics`
14. `ceval_high_school_mathematics`
15. `ceval_high_school_chemistry`
16. `ceval_high_school_physics`

Legacy `4 + 4` benchmark grouping is retained only in `prepare_benchmark_sets.py` for backtracking and result comparison.

## Prompt And Scoring Notes

For `MMLU`:

1. prompt template follows the official `hendrycks/test` `evaluate.py` few-shot structure
2. `dev` split provides the 5 in-context exemplars
3. `validation` split remains the local scored split
4. answer choice scoring uses the appended answer letter tokens (`" A"` / `" B"` / `" C"` / `" D"`)

For `C-Eval`:

1. prompt template follows the official `hkust-nlp/ceval` evaluator few-shot structure
2. `dev` split provides the 5 in-context exemplars
3. `val` split remains the local scored split
4. answer choice scoring uses single answer letters (`"A"` / `"B"` / `"C"` / `"D"`)

## Execution

Run from the repository root:

```bash
/mnt/env/openpangu-publiceval-python.sh experiments/run_final_artifact_benchmark_batch.py /mnt/env/openpangu-publiceval-python.sh
```

## Output Layout

Intermediate run dirs are written to:

```text
/root/final_artifact_benchmark_work/results
```

Per-task run dirs are copied to:

```text
/mnt/openpangu-embedded-7b-model/experiments/results/final_artifact_benchmark/
```

Final summary files are written to:

```text
/mnt/results/final_artifact_benchmark/final_artifact_benchmark_summary.json
/mnt/results/final_artifact_benchmark/final_artifact_benchmark_summary.md
```

At the same time, `build_final_artifact_benchmark_summary.py` mirrors the same two files into the repository-local result entrypoint used by local docs / thesis checks:

```text
/mnt/openpangu-embedded-7b-model/experiments/results/final_artifact_benchmark_summary.json
/mnt/openpangu-embedded-7b-model/experiments/results/final_artifact_benchmark_summary.md
```

## Preparing Benchmark Jsonl Files

If the 16 benchmark jsonl files do not yet exist locally, regenerate them with:

```bash
/mnt/env/openpangu-publiceval-python.sh experiments/data/prepare_benchmark_sets.py --tasks final_artifact_hard_8x8 --max-samples-per-task 0
```

For a quick smoke check:

```bash
/mnt/env/openpangu-publiceval-python.sh experiments/data/prepare_benchmark_sets.py --tasks hard_smoke --max-samples-per-task 2
```
