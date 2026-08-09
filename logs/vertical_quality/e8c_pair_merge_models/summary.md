# E8c 质量引导实例合并模型比较

- 配对数：8156
- 有效配对数：8027
- 特征数：17
- 锁定部署 seed：42
- 推荐模型：**None**
- 推荐阈值：None

| Model | Pair AUC | Pair AP | Precision | Positive-source recall | Unsafe merge rate | Pass seeds |
|---|---:|---:|---:|---:|---:|---:|
| distance_rule | 0.720212 | 0.446211 | 1.000000 | 0.023256 | 0.000000 | 0/3 |
| logistic_regression | 0.783986 | 0.466935 | 1.000000 | 0.003876 | 0.000000 | 0/3 |
| pair_mlp | 0.794093 | 0.508841 | 1.000000 | 0.016796 | 0.000000 | 0/3 |

## 模型资格

- distance_rule: **False**
- logistic_regression: **False**
- pair_mlp: **False**

## Gate

- at_least_one_deployable_model: **False**
- locked_seed_passed: **False**
- mlp_stability_passed_if_selected: **True**
- passed: **False**

STOP：不要进入 pipeline 合并。
