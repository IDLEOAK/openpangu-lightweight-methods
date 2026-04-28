# Current Result Summary

## Scope

- This snapshot summarizes the current prompt-eval/generation full34 runs across the main methods.
- Public-corpus PPL and benchmark aggregates are maintained in their dedicated summary files.

## Baseline

- evaluation_samples: 16
- evaluation_tokens: 836
- perplexity: 444.424158
- peak_memory_mb: 1587.01

## Method Snapshot

| method | route | key setting | PPL | elapsed_s | peak_memory_mb | generation |
| --- | --- | --- | ---: | ---: | ---: | --- |
| sparsegpt | prune | s=0.3 | 469.213927 | 282.5065 | 2466.29 | 36.3549 tokens/s |
| admm | prune | s=0.3 | 461.248566 | 166.0374 | 4544.4 | 35.8834 tokens/s |
| llm_bip | prune-proxy | s=0.08, group=256 | 1196.787471 | 34.4383 | 1586.71 | -- |
| gptq | quant | 4 bit, g128 | 514.224123 | 366.4168 | 2866.3 | 34.4346 tokens/s |
| awq | quant | 4 bit, g128 | 502.081605 | 52.4527 | 1620.76 | 34.5933 tokens/s |
| smoothquant | quant | 8 bit, g128, alpha=0.5 | 453.036672 | 18.2084 | 1588.36 | 34.8843 tokens/s |

## Run Directories

- sparsegpt: `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\sparsegpt\20260402-031540-sparsegpt_port_full34_formal_local`
- admm: `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\admm\20260402-234619-admm_port_full34_formal_local`
- llm_bip: `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\llm_bip\20260403-000122-llm_bip_port_full34_formal_local`
- gptq: `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\gptq\20260402-034750-gptq_port_full34_generation`
- awq: `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\awq\20260402-235228-awq_port_full34_generation`
- smoothquant: `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\smoothquant\20260402-235738-smoothquant_port_full34_generation`
