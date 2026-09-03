# Q6a 欠分割碎片受限合并 Oracle

- 数据：固定五个 validation forests；未读取 Wytham。
- 邻接图只由预测结果生成；固定邻接门限位于实例几何与 base-vote 空间。
- GT 只用于固定 Q3 fragmentation 目标并执行安全 Oracle 验收。
- Fragmentation 目标树：41
- GT-free 候选边：2,480
- 图覆盖目标树：33

## 汇总指标

| Mode | TP | FP | FN | Completeness | Commission | F1 | Recovered | Lost | F1 gain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 1633 | 275 | 149 | 91.639% | 14.413% | 88.509% | - | - | - |
| fragment_union_ceiling | 1641 | 255 | 141 | 92.088% | 13.449% | 89.233% | 8 | 0 | +0.724 pp |
| fragment_graph_oracle | 1642 | 252 | 140 | 92.144% | 13.305% | 89.336% | 9 | 0 | +0.827 pp |

## 每森林图 Oracle

| Plot | Targets | Covered | Recovered | Lost | F1 gain |
|---|---:|---:|---:|---:|---:|
| G4N | 7 | 3 | 0 | 0 | +0.000 pp |
| G4W | 9 | 5 | 0 | 0 | +0.000 pp |
| L1N | 10 | 10 | 3 | 0 | +1.168 pp |
| O1N | 0 | 0 | 0 | 0 | +0.000 pp |
| O1W | 15 | 15 | 6 | 0 | +3.188 pp |

## Gate

- expected_validation_plots: **True**
- baseline_reference_per_plot_compatible: **True**
- baseline_reference_total_tp_drift_passed: **True**
- baseline_reference_total_fp_drift_passed: **True**
- fragmentation_targets_reproduced: **True**
- union_ceiling_passed: **False**
- graph_target_coverage_passed: **True**
- graph_recovery_passed: **False**
- graph_f1_gain_passed: **True**
- graph_lost_tree_passed: **True**
- graph_commission_passed: **True**
- graph_plot_consistency_passed: **True**
- passed: **False**

- 推荐路线：**close_fragment_merge_route**

STOP：关闭 fragmentation 合并路线，不使用 Wytham 调参。
