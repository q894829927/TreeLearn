# 实例可靠性 MLP 控制实验

- 有监督实例：699
- TP：568
- FP：131
- 特征数：35
- 重复分层折数：3 × 5

| 模型 | ROC-AUC | TP AP | FP AP |
|---|---:|---:|---:|
| nested_single_feature | 0.853495 ± 0.029293 | 0.956550 ± 0.010126 | 0.619238 ± 0.075448 |
| logistic_regression | 0.904932 ± 0.019206 | 0.974828 ± 0.006460 | 0.742811 ± 0.050152 |
| instance_mlp | 0.896874 ± 0.034850 | 0.970930 ± 0.015836 | 0.712225 ± 0.067375 |

## Gate

- mlp_auc_at_least_threshold: **True**
- mlp_gain_over_nested_single_passed: **True**
- mlp_not_worse_than_logistic: **False**
- passed: **False**

## 单特征在训练折中的选择次数

```json
{
  "confidence_p90": 2,
  "point_density_bbox": 13
}
```
