# E1b 冻结主干双头 Seed-MLP 控制实验

- 候选总数：6,445,941
- 训练候选：4,537,796
- 验证候选：1,908,145
- 输入维度：61
- 随机种子：[42, 43, 44]
- 锁定部署 seed：42
- 固定 keep ratio：0.780
- Coverage 保护比例：0.100

| Metric | Mean | Std | Min | Max |
|---|---:|---:|---:|---:|
| reliability_roc_auc | 0.733786 | 0.001693 | 0.732011 | 0.735384 |
| reliability_ap | 0.554706 | 0.000899 | 0.554122 | 0.555740 |
| coverage_roc_auc | 0.739902 | 0.001397 | 0.739081 | 0.741516 |
| coverage_ap | 0.142973 | 0.002277 | 0.140769 | 0.145317 |
| dual_critical_recall | 0.864393 | 0.005692 | 0.857917 | 0.868604 |
| dual_critical_tree_coverage | 0.994878 | 0.000569 | 0.994308 | 0.995447 |
| dual_selected_mean_utility | 0.407711 | 0.001167 | 0.406515 | 0.408846 |
| dual_selected_non_tree_rate | 0.075906 | 0.000763 | 0.075310 | 0.076765 |

## 相对控制组

- Critical recall 相对随机：+0.085001
- Critical recall 相对 Reliability-only：+0.048789
- Selected utility 相对随机提升：+16.549%
- Coverage AP / prevalence：2.981x

## Gate

- reliability_auc_passed: **False**
- coverage_ap_lift_passed: **True**
- critical_gain_over_random_passed: **True**
- coverage_head_gain_passed: **True**
- utility_gain_passed: **True**
- critical_tree_coverage_passed: **True**
- per_plot_coverage_passed: **True**
- seed_stability_passed: **True**
- locked_seed_present: **True**
- passed: **False**

STOP：先重新设计种子特征或 Coverage target，不进入注意力。
