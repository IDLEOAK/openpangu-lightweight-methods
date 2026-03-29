# 当前结果汇总

## 统一口径

- 评测样本数：8
- 评测 token 数：458
- baseline perplexity：343.118887

## 当前阶段结果表

| 方案 | 当前设置 | PPL | 运行耗时/s | 峰值显存/MB | 备注 |
| --- | --- | ---: | ---: | ---: | --- |
| Baseline | FP16/BF16 基线 | 343.118887 | 33.5271 | 1587.07 | 当前统一评测口径 |
| SparseGPT | 30% 稀疏，full34 | 342.233663 | 606.8277 | 2466.29 | generation: 跳过（本地 full34 generation OOM） |
| GPTQ | 4-bit, g128, full34 | 386.49593 | 771.2611 | 2866.3 | 当前为 dense 回写路径 |

## 原始结果目录

- SparseGPT: `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\sparsegpt\20260329-171619-sparsegpt_port_full34_generation`
- GPTQ: `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\gptq\20260329-182405-gptq_port_full34_stable`
