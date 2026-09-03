# E2a Cellwise Vote-Refinement Oracle

- 数据：新生成的固定 validation artifacts；未读取 Wytham。
- 不删除种子，不修改 HDBSCAN 参数。
- Vote cell：0.600 m
- 候选种子：1,908,145
- 树种子：1,613,607

## Macro 指标

| Mode | Mean error | Median error | P90 error | Cell purity | Collision | Cluster F1 | Tree recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 0.621051 m | 0.408785 m | 1.283243 m | 97.759% | 4.192% | 79.577% | 96.698% |
| global_shared | 0.612070 m | 0.388992 m | 1.295578 m | 97.788% | 4.403% | 79.577% | 96.698% |
| cellwise_shared | 0.091252 m | 0.073751 m | 0.157126 m | 97.221% | 6.055% | 80.785% | 97.139% |
| pointwise_oracle | 0.000001 m | 0.000001 m | 0.000003 m | 99.619% | 0.171% | 76.981% | 98.555% |

## 效果

- mean_error_reduction_relative: +0.853068
- p90_error_reduction_relative: +0.877556
- pointwise_improvement_retained: +0.853070
- cell_purity_gain: -0.005376
- multi_tree_collision_increase: +0.018635
- plot_wins: +4.000000
- max_pointwise_oracle_mean_error: +0.000002
- cluster_f1_gain_pp: +1.208145
- cluster_tree_recall_drop_pp: -0.440661
- cluster_commission_reduction_pp: +1.463248

## 每森林

| Plot | Seeds | Mean reduction | P90 reduction | F1 gain | Recall drop | Win |
|---|---:|---:|---:|---:|---:|---|
| G4N | 293,705 | +89.531% | +91.442% | +1.304 pp | -0.984 pp | True |
| G4W | 519,142 | +85.591% | +89.071% | +1.164 pp | -0.520 pp | True |
| L1N | 452,316 | +83.447% | +86.500% | +2.464 pp | -0.442 pp | True |
| O1N | 242,281 | +83.036% | +83.321% | -1.146 pp | +1.258 pp | False |
| O1W | 400,701 | +82.763% | +85.412% | +2.256 pp | -1.515 pp | True |

## Gate

- expected_validation_plots_present: **True**
- exact_pointwise_ceiling_passed: **True**
- mean_error_reduction_passed: **True**
- p90_error_reduction_passed: **True**
- pointwise_improvement_retained: **True**
- cell_purity_not_worse: **False**
- collision_not_worse: **False**
- per_plot_consistency_passed: **True**
- cluster_f1_gain_passed: **True**
- cluster_tree_recall_preserved: **True**
- passed: **False**

STOP：关闭 Cellwise vote-refinement，不读取 Wytham 调参。
