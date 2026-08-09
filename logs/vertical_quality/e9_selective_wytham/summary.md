# E9 选择性实例分割：wytham_final

- 数据角色：**final_evaluation_no_tuning**
- 原始实例标签保持不变；本阶段只评估质量排序。
- 实例数：1862（TP 568，counted FP 131，ignored 1163）
- 随机排序对照：500 次，seed 20260804

## 面积指标

| Metric | Vertical-MLP | Random | Delta |
|---|---:|---:|---:|
| Commission area（越低越好） | 10.989% | 18.728% | 41.322% relative reduction |
| Detection F1 area（越高越好） | 70.562% | 60.269% | +10.293 pp |

## 关键保留率

| Retained | Score F1 | Random F1 | Score Commission | Random Commission | Score Completeness |
|---:|---:|---:|---:|---:|---:|
| 50.000% | 62.643% | 46.295% | 5.093% | 18.673% | 46.750% |
| 75.027% | 72.393% | 60.803% | 10.000% | 18.740% | 60.547% |
| 85.016% | 73.232% | 65.660% | 12.893% | 18.738% | 63.170% |
| 100.000% | 72.081% | 72.081% | 18.741% | 18.741% | 64.766% |

## Gate

- score_only_metadata_passed: **True**
- baseline_counts_reproduced: **True**
- commission_area_improved: **True**
- f1_area_improved: **True**
- passed: **True**

PASS：质量分数提供了优于随机排序的选择性风险控制。
