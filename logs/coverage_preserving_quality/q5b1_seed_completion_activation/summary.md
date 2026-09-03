# Q5b1 Seed-Completion proposal activation

- Uses only the fixed train/validation forests; Wytham is forbidden.
- Recommended model: **None**
- Locked seed: 42

| Model | Runs | Pass | Activation AP | Target recall | Eligible |
|---|---:|---:|---:|---:|---|
| multitask_mlp | 3 | 0/2 | 0.048666 | 72.043% | False |
| logistic_regression | 1 | 0/1 | 0.019734 | 93.548% | False |
| fixed_rule | 1 | 0/1 | 0.017625 | 51.613% | False |

## Gate

- data_gate_passed: **True**
- at_least_one_eligible_model: **False**
- locked_seed_passed: **False**
- passed: **False**

STOP: proposal activation is not deployable; close Seed-Completion.
