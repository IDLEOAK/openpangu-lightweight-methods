# 当前结果汇总

## 统一口径

- 评测样本数：16
- 评测 token 数：836
- baseline perplexity：444.424158

## 当前阶段结果表

| 方案 | 当前设置 | PPL | 运行耗时/s | 峰值显存/MB | 备注 |
| --- | --- | ---: | ---: | ---: | --- |
| Baseline | FP16/BF16 基线 | 444.424158 | 107.9017 | 1587.01 | 当前统一评测口径 |
| SparseGPT | 30% 稀疏，full34 | 469.213927 | 282.5065 | 2466.29 | generation: 36.3549 tokens/s |
| GPTQ | 4-bit, g128, full34 | 514.224123 | 366.4168 | 2866.3 | generation: 34.4346 tokens/s |

## 原始结果目录

- SparseGPT: `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\sparsegpt\20260402-031540-sparsegpt_port_full34_formal_local`
- GPTQ: `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\gptq\20260402-034750-gptq_port_full34_generation`
