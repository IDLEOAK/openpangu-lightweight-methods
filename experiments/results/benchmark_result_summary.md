# Benchmark Result Summary

## Aggregate Metrics

Macro average is computed by giving each task equal weight within the selected task set.
Weighted average is computed by weighting each task by its evaluated sample count.

| method | macro(all 15) | weighted(all 571) | macro(en 8) | weighted(en 195) | macro(zh 7) | weighted(zh 376) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.365664 | 0.353765 | 0.412938 | 0.466667 | 0.311636 | 0.295213 |
| sparsegpt | 0.35189 | 0.366025 | 0.388551 | 0.482051 | 0.309992 | 0.305851 |
| admm | 0.297162 | 0.30648 | 0.341904 | 0.415385 | 0.246028 | 0.25 |
| gptq | 0.351708 | 0.350263 | 0.388195 | 0.45641 | 0.310007 | 0.295213 |
| awq | 0.306954 | 0.325744 | 0.335737 | 0.430769 | 0.274058 | 0.271277 |
| smoothquant | 0.37229 | 0.367776 | 0.412938 | 0.466667 | 0.325836 | 0.316489 |

## boolq_validation_mcq

| method | accuracy | correct/evaluated | run_dir |
| --- | ---: | ---: | --- |
| baseline | 0.75 | 48/64 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\benchmark\20260407-000535-benchmark_public_mcq_baseline_boolq_validation_mcq_boolq_validation_mcq` |
| sparsegpt | 0.796875 | 51/64 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\sparsegpt\20260407-000821-sparsegpt_port_full34_benchmark_boolq_validation_mcq` |
| admm | 0.671875 | 43/64 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\admm\20260407-001552-admm_port_full34_benchmark_boolq_validation_mcq` |
| gptq | 0.71875 | 46/64 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\gptq\20260407-001945-gptq_port_full34_benchmark_boolq_validation_mcq` |
| awq | 0.703125 | 45/64 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\awq\20260407-002842-awq_port_full34_benchmark_boolq_validation_mcq` |
| smoothquant | 0.75 | 48/64 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\smoothquant\20260407-003216-smoothquant_port_full34_benchmark_boolq_validation_mcq` |

## ceval_college_programming_val_mcq

| method | accuracy | correct/evaluated | run_dir |
| --- | ---: | ---: | --- |
| baseline | 0.27027 | 10/37 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\benchmark\20260407-003500-benchmark_public_mcq_baseline_ceval_college_programming_val_mcq_ceval_college_programming_val_mcq` |
| sparsegpt | 0.27027 | 10/37 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\sparsegpt\20260407-003731-sparsegpt_port_full34_benchmark_ceval_college_programming_val_mcq` |
| admm | 0.162162 | 6/37 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\admm\20260407-004457-admm_port_full34_benchmark_ceval_college_programming_val_mcq` |
| gptq | 0.243243 | 9/37 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\gptq\20260407-004847-gptq_port_full34_benchmark_ceval_college_programming_val_mcq` |
| awq | 0.243243 | 9/37 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\awq\20260407-005740-awq_port_full34_benchmark_ceval_college_programming_val_mcq` |
| smoothquant | 0.297297 | 11/37 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\smoothquant\20260407-010113-smoothquant_port_full34_benchmark_ceval_college_programming_val_mcq` |

## ceval_computer_network_val_mcq

| method | accuracy | correct/evaluated | run_dir |
| --- | ---: | ---: | --- |
| baseline | 0.473684 | 9/19 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\benchmark\20260407-010356-benchmark_public_mcq_baseline_ceval_computer_network_val_mcq_ceval_computer_network_val_mcq` |
| sparsegpt | 0.368421 | 7/19 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\sparsegpt\20260407-010620-sparsegpt_port_full34_benchmark_ceval_computer_network_val_mcq` |
| admm | 0.263158 | 5/19 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\admm\20260407-011343-admm_port_full34_benchmark_ceval_computer_network_val_mcq` |
| gptq | 0.473684 | 9/19 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\gptq\20260407-011730-gptq_port_full34_benchmark_ceval_computer_network_val_mcq` |
| awq | 0.315789 | 6/19 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\awq\20260407-012616-awq_port_full34_benchmark_ceval_computer_network_val_mcq` |
| smoothquant | 0.421053 | 8/19 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\smoothquant\20260407-012948-smoothquant_port_full34_benchmark_ceval_computer_network_val_mcq` |

## cmmlu_college_mathematics_train_mcq

| method | accuracy | correct/evaluated | run_dir |
| --- | ---: | ---: | --- |
| baseline | 0.28125 | 18/64 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\benchmark\20260407-013226-benchmark_public_mcq_baseline_cmmlu_college_mathematics_train_mcq_cmmlu_college_mathematics_train_mcq` |
| sparsegpt | 0.296875 | 19/64 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\sparsegpt\20260407-013456-sparsegpt_port_full34_benchmark_cmmlu_college_mathematics_train_mcq` |
| admm | 0.234375 | 15/64 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\admm\20260407-014226-admm_port_full34_benchmark_cmmlu_college_mathematics_train_mcq` |
| gptq | 0.234375 | 15/64 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\gptq\20260407-014624-gptq_port_full34_benchmark_cmmlu_college_mathematics_train_mcq` |
| awq | 0.25 | 16/64 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\awq\20260407-015524-awq_port_full34_benchmark_cmmlu_college_mathematics_train_mcq` |
| smoothquant | 0.296875 | 19/64 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\smoothquant\20260407-015904-smoothquant_port_full34_benchmark_cmmlu_college_mathematics_train_mcq` |

## cmmlu_computer_science_train_mcq

| method | accuracy | correct/evaluated | run_dir |
| --- | ---: | ---: | --- |
| baseline | 0.3125 | 20/64 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\benchmark\20260407-020155-benchmark_public_mcq_baseline_cmmlu_computer_science_train_mcq_cmmlu_computer_science_train_mcq` |
| sparsegpt | 0.359375 | 23/64 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\sparsegpt\20260407-020425-sparsegpt_port_full34_benchmark_cmmlu_computer_science_train_mcq` |
| admm | 0.265625 | 17/64 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\admm\20260407-021157-admm_port_full34_benchmark_cmmlu_computer_science_train_mcq` |
| gptq | 0.34375 | 22/64 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\gptq\20260407-021553-gptq_port_full34_benchmark_cmmlu_computer_science_train_mcq` |
| awq | 0.21875 | 14/64 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\awq\20260407-022450-awq_port_full34_benchmark_cmmlu_computer_science_train_mcq` |
| smoothquant | 0.34375 | 22/64 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\smoothquant\20260407-022828-smoothquant_port_full34_benchmark_cmmlu_computer_science_train_mcq` |

## cmmlu_computer_security_train_mcq

| method | accuracy | correct/evaluated | run_dir |
| --- | ---: | ---: | --- |
| baseline | 0.359375 | 23/64 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\benchmark\20260407-023117-benchmark_public_mcq_baseline_cmmlu_computer_security_train_mcq_cmmlu_computer_security_train_mcq` |
| sparsegpt | 0.390625 | 25/64 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\sparsegpt\20260407-023347-sparsegpt_port_full34_benchmark_cmmlu_computer_security_train_mcq` |
| admm | 0.296875 | 19/64 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\admm\20260407-024120-admm_port_full34_benchmark_cmmlu_computer_security_train_mcq` |
| gptq | 0.265625 | 17/64 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\gptq\20260407-024516-gptq_port_full34_benchmark_cmmlu_computer_security_train_mcq` |
| awq | 0.375 | 24/64 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\awq\20260407-025444-awq_port_full34_benchmark_cmmlu_computer_security_train_mcq` |
| smoothquant | 0.390625 | 25/64 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\smoothquant\20260407-025823-smoothquant_port_full34_benchmark_cmmlu_computer_security_train_mcq` |

## cmmlu_high_school_mathematics_train_mcq

| method | accuracy | correct/evaluated | run_dir |
| --- | ---: | ---: | --- |
| baseline | 0.203125 | 13/64 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\benchmark\20260407-030110-benchmark_public_mcq_baseline_cmmlu_high_school_mathematics_train_mcq_cmmlu_high_school_mathematics_train_mcq` |
| sparsegpt | 0.234375 | 15/64 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\sparsegpt\20260407-030340-sparsegpt_port_full34_benchmark_cmmlu_high_school_mathematics_train_mcq` |
| admm | 0.265625 | 17/64 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\admm\20260407-031120-admm_port_full34_benchmark_cmmlu_high_school_mathematics_train_mcq` |
| gptq | 0.171875 | 11/64 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\gptq\20260407-031516-gptq_port_full34_benchmark_cmmlu_high_school_mathematics_train_mcq` |
| awq | 0.234375 | 15/64 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\awq\20260407-032411-awq_port_full34_benchmark_cmmlu_high_school_mathematics_train_mcq` |
| smoothquant | 0.203125 | 13/64 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\smoothquant\20260407-032752-smoothquant_port_full34_benchmark_cmmlu_high_school_mathematics_train_mcq` |

## cmmlu_machine_learning_train_mcq

| method | accuracy | correct/evaluated | run_dir |
| --- | ---: | ---: | --- |
| baseline | 0.28125 | 18/64 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\benchmark\20260407-033041-benchmark_public_mcq_baseline_cmmlu_machine_learning_train_mcq_cmmlu_machine_learning_train_mcq` |
| sparsegpt | 0.25 | 16/64 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\sparsegpt\20260407-033311-sparsegpt_port_full34_benchmark_cmmlu_machine_learning_train_mcq` |
| admm | 0.234375 | 15/64 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\admm\20260407-034042-admm_port_full34_benchmark_cmmlu_machine_learning_train_mcq` |
| gptq | 0.4375 | 28/64 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\gptq\20260407-034439-gptq_port_full34_benchmark_cmmlu_machine_learning_train_mcq` |
| awq | 0.28125 | 18/64 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\awq\20260407-035334-awq_port_full34_benchmark_cmmlu_machine_learning_train_mcq` |
| smoothquant | 0.328125 | 21/64 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\smoothquant\20260407-035714-smoothquant_port_full34_benchmark_cmmlu_machine_learning_train_mcq` |

## hellaswag_validation_mcq

| method | accuracy | correct/evaluated | run_dir |
| --- | ---: | ---: | --- |
| baseline | 0.265625 | 17/64 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\benchmark\20260407-040002-benchmark_public_mcq_baseline_hellaswag_validation_mcq_hellaswag_validation_mcq` |
| sparsegpt | 0.328125 | 21/64 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\sparsegpt\20260407-040233-sparsegpt_port_full34_benchmark_hellaswag_validation_mcq` |
| admm | 0.28125 | 18/64 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\admm\20260407-041003-admm_port_full34_benchmark_hellaswag_validation_mcq` |
| gptq | 0.3125 | 20/64 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\gptq\20260407-041358-gptq_port_full34_benchmark_hellaswag_validation_mcq` |
| awq | 0.3125 | 20/64 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\awq\20260407-042257-awq_port_full34_benchmark_hellaswag_validation_mcq` |
| smoothquant | 0.265625 | 17/64 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\smoothquant\20260407-042637-smoothquant_port_full34_benchmark_hellaswag_validation_mcq` |

## mmlu_college_computer_science_validation_mcq

| method | accuracy | correct/evaluated | run_dir |
| --- | ---: | ---: | --- |
| baseline | 0.636364 | 7/11 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\benchmark\20260407-042926-benchmark_public_mcq_baseline_mmlu_college_computer_science_validation_mcq_mmlu_college_computer_science_validation_mcq` |
| sparsegpt | 0.545455 | 6/11 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\sparsegpt\20260407-043149-sparsegpt_port_full34_benchmark_mmlu_college_computer_science_validation_mcq` |
| admm | 0.272727 | 3/11 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\admm\20260407-043907-admm_port_full34_benchmark_mmlu_college_computer_science_validation_mcq` |
| gptq | 0.363636 | 4/11 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\gptq\20260407-044251-gptq_port_full34_benchmark_mmlu_college_computer_science_validation_mcq` |
| awq | 0.363636 | 4/11 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\awq\20260407-045138-awq_port_full34_benchmark_mmlu_college_computer_science_validation_mcq` |
| smoothquant | 0.636364 | 7/11 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\smoothquant\20260407-045505-smoothquant_port_full34_benchmark_mmlu_college_computer_science_validation_mcq` |

## mmlu_college_mathematics_validation_mcq

| method | accuracy | correct/evaluated | run_dir |
| --- | ---: | ---: | --- |
| baseline | 0.272727 | 3/11 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\benchmark\20260407-045741-benchmark_public_mcq_baseline_mmlu_college_mathematics_validation_mcq_mmlu_college_mathematics_validation_mcq` |
| sparsegpt | 0.181818 | 2/11 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\sparsegpt\20260407-050004-sparsegpt_port_full34_benchmark_mmlu_college_mathematics_validation_mcq` |
| admm | 0.272727 | 3/11 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\admm\20260407-050726-admm_port_full34_benchmark_mmlu_college_mathematics_validation_mcq` |
| gptq | 0.363636 | 4/11 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\gptq\20260407-051112-gptq_port_full34_benchmark_mmlu_college_mathematics_validation_mcq` |
| awq | 0.272727 | 3/11 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\awq\20260407-051958-awq_port_full34_benchmark_mmlu_college_mathematics_validation_mcq` |
| smoothquant | 0.272727 | 3/11 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\smoothquant\20260407-052326-smoothquant_port_full34_benchmark_mmlu_college_mathematics_validation_mcq` |

## mmlu_computer_security_validation_mcq

| method | accuracy | correct/evaluated | run_dir |
| --- | ---: | ---: | --- |
| baseline | 0.363636 | 4/11 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\benchmark\20260407-052602-benchmark_public_mcq_baseline_mmlu_computer_security_validation_mcq_mmlu_computer_security_validation_mcq` |
| sparsegpt | 0.090909 | 1/11 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\sparsegpt\20260407-052827-sparsegpt_port_full34_benchmark_mmlu_computer_security_validation_mcq` |
| admm | 0.090909 | 1/11 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\admm\20260407-053542-admm_port_full34_benchmark_mmlu_computer_security_validation_mcq` |
| gptq | 0.181818 | 2/11 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\gptq\20260407-053927-gptq_port_full34_benchmark_mmlu_computer_security_validation_mcq` |
| awq | 0.090909 | 1/11 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\awq\20260407-054812-awq_port_full34_benchmark_mmlu_computer_security_validation_mcq` |
| smoothquant | 0.363636 | 4/11 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\smoothquant\20260407-055140-smoothquant_port_full34_benchmark_mmlu_computer_security_validation_mcq` |

## mmlu_formal_logic_validation_mcq

| method | accuracy | correct/evaluated | run_dir |
| --- | ---: | ---: | --- |
| baseline | 0.5 | 7/14 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\benchmark\20260407-055416-benchmark_public_mcq_baseline_mmlu_formal_logic_validation_mcq_mmlu_formal_logic_validation_mcq` |
| sparsegpt | 0.357143 | 5/14 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\sparsegpt\20260407-055640-sparsegpt_port_full34_benchmark_mmlu_formal_logic_validation_mcq` |
| admm | 0.428571 | 6/14 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\admm\20260407-060358-admm_port_full34_benchmark_mmlu_formal_logic_validation_mcq` |
| gptq | 0.357143 | 5/14 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\gptq\20260407-060745-gptq_port_full34_benchmark_mmlu_formal_logic_validation_mcq` |
| awq | 0.357143 | 5/14 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\awq\20260407-061633-awq_port_full34_benchmark_mmlu_formal_logic_validation_mcq` |
| smoothquant | 0.5 | 7/14 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\smoothquant\20260407-062004-smoothquant_port_full34_benchmark_mmlu_formal_logic_validation_mcq` |

## mmlu_high_school_computer_science_validation_mcq

| method | accuracy | correct/evaluated | run_dir |
| --- | ---: | ---: | --- |
| baseline | 0.333333 | 3/9 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\benchmark\20260407-062242-benchmark_public_mcq_baseline_mmlu_high_school_computer_science_validation_mcq_mmlu_high_school_computer_science_validation_mcq` |
| sparsegpt | 0.444444 | 4/9 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\sparsegpt\20260407-062508-sparsegpt_port_full34_benchmark_mmlu_high_school_computer_science_validation_mcq` |
| admm | 0.444444 | 4/9 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\admm\20260407-063230-admm_port_full34_benchmark_mmlu_high_school_computer_science_validation_mcq` |
| gptq | 0.444444 | 4/9 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\gptq\20260407-063616-gptq_port_full34_benchmark_mmlu_high_school_computer_science_validation_mcq` |
| awq | 0.222222 | 2/9 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\awq\20260407-064500-awq_port_full34_benchmark_mmlu_high_school_computer_science_validation_mcq` |
| smoothquant | 0.333333 | 3/9 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\smoothquant\20260407-064829-smoothquant_port_full34_benchmark_mmlu_high_school_computer_science_validation_mcq` |

## mmlu_machine_learning_validation_mcq

| method | accuracy | correct/evaluated | run_dir |
| --- | ---: | ---: | --- |
| baseline | 0.181818 | 2/11 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\benchmark\20260407-065106-benchmark_public_mcq_baseline_mmlu_machine_learning_validation_mcq_mmlu_machine_learning_validation_mcq` |
| sparsegpt | 0.363636 | 4/11 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\sparsegpt\20260407-065332-sparsegpt_port_full34_benchmark_mmlu_machine_learning_validation_mcq` |
| admm | 0.272727 | 3/11 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\admm\20260407-070052-admm_port_full34_benchmark_mmlu_machine_learning_validation_mcq` |
| gptq | 0.363636 | 4/11 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\gptq\20260407-070440-gptq_port_full34_benchmark_mmlu_machine_learning_validation_mcq` |
| awq | 0.363636 | 4/11 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\awq\20260407-071326-awq_port_full34_benchmark_mmlu_machine_learning_validation_mcq` |
| smoothquant | 0.181818 | 2/11 | `C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\smoothquant\20260407-071657-smoothquant_port_full34_benchmark_mmlu_machine_learning_validation_mcq` |

