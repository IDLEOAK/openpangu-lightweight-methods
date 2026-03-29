# 当前结果汇总

## 统一口径

- 评测样本数：16
- 评测 token 数：836
- baseline perplexity：453.479042

## 当前阶段结果表

| 方案 | 当前设置 | PPL | 运行耗时/s | 峰值显存/MB | 备注 |
| --- | --- | ---: | ---: | ---: | --- |
| Baseline | FP16/BF16 基线 | 453.479042 | 33.7329 | 1587.01 | 当前统一评测口径 |
| SparseGPT | 30% 稀疏，full34 | 482.22591 | 623.2251 | 2466.29 | generation: 跳过（本地 full34 generation OOM） |
| GPTQ | 4-bit, g128, full34 | 489.352636 | 800.7737 | 2866.3 | 当前为 dense 回写路径 |

## 原始结果目录

- SparseGPT: `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\sparsegpt\20260329-193540-sparsegpt_port_full34_formal_local`
- GPTQ: `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\gptq\20260329-195336-gptq_port_full34_formal_local`
