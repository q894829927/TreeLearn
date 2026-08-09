# E9 选择性实例分割：l1w_regression

- 数据角色：**regression_validation**
- 原始实例标签保持不变；本阶段只评估质量排序。
- 实例数：371（TP 156，counted FP 5，ignored 210）
- 随机排序对照：500 次，seed 20260804

## 面积指标

| Metric | Vertical-MLP | Random | Delta |
|---|---:|---:|---:|
| Commission area（越低越好） | 0.678% | 3.060% | 77.855% relative reduction |
| Detection F1 area（越高越好） | 94.708% | 83.746% | +10.962 pp |

## 关键保留率

| Retained | Score F1 | Random F1 | Score Commission | Random Commission | Score Completeness |
|---:|---:|---:|---:|---:|---:|
| 50.135% | 80.916% | 66.025% | 0.000% | 3.011% | 67.949% |
| 75.202% | 99.678% | 84.521% | 0.000% | 3.083% | 99.359% |
| 85.175% | 99.681% | 90.632% | 0.637% | 3.067% | 100.000% |
| 100.000% | 98.423% | 98.423% | 3.106% | 3.106% | 100.000% |

## Gate

- score_only_metadata_passed: **True**
- baseline_counts_reproduced: **True**
- commission_area_improved: **True**
- f1_area_improved: **True**
- passed: **True**

PASS：质量分数提供了优于随机排序的选择性风险控制。
