# E1c 参数匹配的 Complexity Seed Relation Attention

- Seeds：[42, 43, 44]
- 锁定 seed：42
- K / radius：8 / 0.600 m
- 参数差比例：3.554%
- Attention tree-AUC seed wins：1/3

| Model | Params | Reliability AUC | Tree-only AUC | Coverage AP | Critical recall | Utility |
|---|---:|---:|---:|---:|---:|---:|
| neighborhood_mlp | 52226 | 0.739388 | 0.668901 | 0.175286 | 0.871853 | 0.408159 |
| relation_attention | 50370 | 0.738234 | 0.667910 | 0.172763 | 0.876347 | 0.407707 |

## Attention − Neighborhood-MLP

- reliability_roc_auc: -0.001154
- tree_only_roc_auc: -0.000990
- coverage_ap: -0.002524
- critical_recall: +0.004495
- selected_utility_relative: -0.001108

## Gate

- neighbor_data_gate_passed: **True**
- parameter_count_matched: **True**
- absolute_reliability_auc_passed: **False**
- absolute_tree_only_auc_passed: **False**
- reliability_gain_passed: **False**
- tree_only_gain_passed: **False**
- coverage_not_worse: **True**
- critical_recall_not_worse: **True**
- selected_utility_gain_passed: **False**
- seed_wins_passed: **False**
- attention_stability_passed: **True**
- locked_seed_present: **True**
- passed: **False**

STOP：注意力未稳定优于参数匹配的邻域 MLP，不进入 pipeline。
