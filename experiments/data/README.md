# Public Eval Data

This directory now contains two kinds of evaluation data:

1. project-specific prompt sets
   - `demo_prompts.jsonl`
   - `formal_eval_prompts.jsonl`
2. public supplemental evaluation corpora
   - `wikitext2_test_public_eval.jsonl`
   - `chinesewebtext2_test_public_eval.jsonl`
3. public multiple-choice benchmark sets
   - `benchmarks/*.jsonl`

## Public corpora choice

The public supplemental corpora are prepared as follows:

1. English public corpus:
   - source: `wikitext` / `wikitext-2-raw-v1` / `test`
   - export rule: keep non-empty rows with at least 5 whitespace-separated words, and skip section-title rows starting with `=`
2. Chinese public corpus:
   - source: `CASIA-LM/ChineseWebText2.0` / `test`
   - export rule: keep non-empty rows with at least 20 characters and `quality_score >= 0.5`

These corpora are not used for calibration.

They are only used as public supplemental perplexity evaluation sets to reduce the dependence on project-specific explanatory prompts.

## Export command

```powershell
C:\Tools\anaconda3\python.exe experiments\data\prepare_public_eval_sets.py
```

## Notes

1. `wikitext2_test_public_eval.jsonl` and `chinesewebtext2_test_public_eval.jsonl` are plain `jsonl` files with a `text` field.
2. The current experiment framework reads them through `experiments.common.data.load_text_samples`.
3. Public-corpus evaluation configs should keep calibration on `demo_prompts.jsonl` and switch only `evaluation_data.path`.
4. Benchmark jsonl files store `prompt`, `choices`, and `answer_index` for multiple-choice evaluation.
5. The first benchmark task set is exported by:

```powershell
C:\Tools\anaconda3\python.exe experiments\data\prepare_benchmark_sets.py --max-samples-per-task 128
```

6. Supported benchmark task groups:
   - `smoke`: quick local validation
   - `mmlu_core_en`: English MMLU core subjects
   - `cmmlu_core_zh`: Chinese CMMLU core subjects
   - `formal_core`: the current formal benchmark candidate set
7. `cmmlu_core_zh` currently uses `svjack/cmmlu` `train` split filtered by subject because it is the most stable loadable Chinese MMLU-style source in the current environment. This should be stated explicitly when writing up results.
