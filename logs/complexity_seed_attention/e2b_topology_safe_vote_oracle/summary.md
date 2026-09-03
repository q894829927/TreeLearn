# E2b 拓扑安全门控 Vote-Refinement Oracle

- 数据：固定五个 validation forests；未读取 Wytham。
- Cell size、候选种子和 HDBSCAN 参数与 E2a 完全一致。
- E2a 的失败结论保持不变；本阶段是新的安全门控假设。
- 候选种子：1,908,145
- 树种子：1,613,607
- 安全 Cell 比例：2.752%
- 被修正树种子比例：4.825%

## Macro 指标

| Mode | Mean error | P90 error | Cell purity | Collision | Cluster F1 | Commission | Tree recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 0.621051 m | 1.283243 m | 97.759% | 4.192% | 79.577% | 30.199% | 96.698% |
| damped_0p25 | 0.472417 m | 0.967154 m | 97.786% | 4.868% | 79.438% | 30.527% | 97.184% |
| damped_0p50 | 0.326351 m | 0.656587 m | 97.781% | 5.334% | 79.837% | 30.043% | 97.370% |
| damped_0p75 | 0.187763 m | 0.360743 m | 97.574% | 5.720% | 80.452% | 29.369% | 97.556% |
| damped_1p00 | 0.091252 m | 0.157126 m | 97.221% | 6.055% | 80.785% | 28.736% | 97.139% |
| topology_safe | 0.615732 m | 1.283243 m | 97.759% | 4.192% | 77.538% | 33.139% | 96.301% |
| pointwise_oracle | 0.000001 m | 0.000003 m | 99.619% | 0.171% | 76.981% | 35.135% | 98.555% |

## 拓扑安全模式效果

- mean_error_reduction_relative: +0.008564
- p90_error_reduction_relative: +0.000000
- cell_purity_gain: +0.000000
- multi_tree_collision_increase: +0.000000
- max_pointwise_oracle_mean_error: +0.000002
- cluster_f1_gain_pp: -2.038603
- cluster_tree_recall_drop_pp: +0.397307
- cluster_commission_reduction_pp: -2.939597
- nonnegative_f1_plots: +0.000000
- worst_plot_f1_gain_pp: -3.269659

## 每森林

| Plot | Safe cells | Corrected trees | Mean reduction | F1 gain | Commission reduction | Recall drop |
|---|---:|---:|---:|---:|---:|---:|
| G4N | 3.056% | 2.714% | +0.470% | -1.575 pp | -2.622 pp | +0.000 pp |
| G4W | 3.821% | 4.531% | +0.857% | -1.888 pp | -3.290 pp | +0.000 pp |
| L1N | 3.018% | 6.573% | +1.333% | -1.971 pp | -2.582 pp | +0.221 pp |
| O1N | 2.923% | 5.572% | +1.085% | -3.270 pp | -4.962 pp | +0.629 pp |
| O1W | 0.941% | 4.734% | +0.776% | -1.489 pp | -1.243 pp | +1.136 pp |

## Gate

- expected_validation_plots_present: **True**
- exact_pointwise_ceiling_passed: **True**
- mean_error_reduction_passed: **False**
- p90_error_reduction_passed: **False**
- cell_purity_preserved: **True**
- collision_preserved: **True**
- cluster_f1_gain_passed: **False**
- commission_reduction_passed: **False**
- tree_recall_preserved: **True**
- per_plot_nonnegative_passed: **False**
- worst_plot_drop_passed: **False**
- passed: **False**

STOP：关闭 vote-refinement/seed-attention 路线，不使用 Wytham 调参。
