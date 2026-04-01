import json
from pathlib import Path


EXPERIMENTS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENTS_ROOT.parent
RESULTS_ROOT = EXPERIMENTS_ROOT / "results"


def latest_run(method: str, prefix: str) -> Path:
    base = RESULTS_ROOT / method
    candidates = [p for p in base.iterdir() if p.is_dir() and p.name.endswith(prefix)]
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No run found for {method} with suffix {prefix}")
    return candidates[0]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    sparsegpt_dir = latest_run("sparsegpt", "sparsegpt_port_full34_formal_local")
    gptq_dir = latest_run("gptq", "gptq_port_full34_generation")

    sparsegpt_summary = read_json(sparsegpt_dir / "summary.json")
    gptq_summary = read_json(gptq_dir / "summary.json")

    baseline_perplexity = sparsegpt_summary["baseline_perplexity"]["perplexity"]
    baseline_eval_tokens = sparsegpt_summary["baseline_perplexity"]["token_count"]
    baseline_eval_samples = sparsegpt_summary["baseline_perplexity"]["sample_count"]

    payload = {
        "baseline": {
            "perplexity": baseline_perplexity,
            "evaluation_samples": baseline_eval_samples,
            "evaluation_tokens": baseline_eval_tokens,
            "peak_memory_mb": sparsegpt_summary["baseline_perplexity"]["peak_memory_mb"],
        },
        "sparsegpt": {
            "run_dir": str(sparsegpt_dir),
            "sparsity": sparsegpt_summary["sparsegpt_plan"]["sparsity"],
            "layer_count": sparsegpt_summary["prune_stats"]["layer_count"],
            "module_count": sparsegpt_summary["prune_stats"]["module_count"],
            "total_target_params": sparsegpt_summary["prune_stats"]["total_pruned_params"],
            "zero_params": sparsegpt_summary["prune_stats"]["total_zero_params"],
            "zero_fraction": sparsegpt_summary["prune_stats"]["overall_zero_fraction"],
            "perplexity": sparsegpt_summary["pruned_perplexity"]["perplexity"],
            "elapsed_s": sparsegpt_summary["prune_stats"]["elapsed_s"],
            "peak_memory_mb": sparsegpt_summary["prune_stats"]["peak_memory_mb"],
            "generation": sparsegpt_summary.get("pruned_generation", {}),
        },
        "gptq": {
            "run_dir": str(gptq_dir),
            "bits": gptq_summary["gptq_plan"]["bits"],
            "group_size": gptq_summary["gptq_plan"]["group_size"],
            "layer_count": gptq_summary["quant_stats"]["layer_count"],
            "module_count": gptq_summary["quant_stats"]["module_count"],
            "total_target_params": gptq_summary["quant_stats"]["total_target_params"],
            "quantized_fraction": gptq_summary["quant_stats"]["quantized_fraction"],
            "perplexity": gptq_summary["quantized_perplexity"]["perplexity"],
            "elapsed_s": gptq_summary["quant_stats"]["elapsed_s"],
            "peak_memory_mb": gptq_summary["quant_stats"]["peak_memory_mb"],
            "generation": gptq_summary.get("quantized_generation", {}),
        },
    }

    json_path = RESULTS_ROOT / "current_result_summary.json"
    md_path = RESULTS_ROOT / "current_result_summary.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    sparsegpt_generation = payload["sparsegpt"]["generation"]
    if sparsegpt_generation.get("skipped"):
        sparsegpt_generation_text = "跳过（本地 full34 generation OOM）"
    elif sparsegpt_generation:
        sparsegpt_generation_text = str(sparsegpt_generation.get("tokens_per_second"))
    else:
        sparsegpt_generation_text = "未记录"

    gptq_generation = payload["gptq"]["generation"]
    if gptq_generation.get("skipped"):
        gptq_generation_text = "跳过（本地 full34 generation OOM）"
    elif gptq_generation:
        gptq_generation_text = str(gptq_generation.get("tokens_per_second"))
    else:
        gptq_generation_text = "未记录"

    markdown = f"""# 当前结果汇总

## 统一口径

- 评测样本数：{baseline_eval_samples}
- 评测 token 数：{baseline_eval_tokens}
- baseline perplexity：{baseline_perplexity}

## 当前阶段结果表

| 方案 | 当前设置 | PPL | 运行耗时/s | 峰值显存/MB | 备注 |
| --- | --- | ---: | ---: | ---: | --- |
| Baseline | FP16/BF16 基线 | {baseline_perplexity} | {sparsegpt_summary["baseline_perplexity"]["elapsed_s"]} | {sparsegpt_summary["baseline_perplexity"]["peak_memory_mb"]} | 当前统一评测口径 |
| SparseGPT | 30% 稀疏，full34 | {payload["sparsegpt"]["perplexity"]} | {payload["sparsegpt"]["elapsed_s"]} | {payload["sparsegpt"]["peak_memory_mb"]} | generation: {sparsegpt_generation_text} |
| GPTQ | 4-bit, g128, full34 | {payload["gptq"]["perplexity"]} | {payload["gptq"]["elapsed_s"]} | {payload["gptq"]["peak_memory_mb"]} | generation: {gptq_generation_text} |

## 原始结果目录

- SparseGPT: `{payload["sparsegpt"]["run_dir"]}`
- GPTQ: `{payload["gptq"]["run_dir"]}`
"""
    md_path.write_text(markdown, encoding="utf-8")
    print(f"[OK] json={json_path}")
    print(f"[OK] markdown={md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
