# E10b Global + Vertical 互补性验证

- 实例数：9402
- Global-Wide 参数：11,394
- Fusion 参数：11,842
- 参数差比例：3.783%

| Model | FP AP | IoU MAE | ROC-AUC | Spearman |
|---|---:|---:|---:|---:|
| global_wide | 0.988111 | 0.090398 | 0.992709 | 0.822146 |
| global_vertical_fusion | 0.992072 | 0.088352 | 0.995195 | 0.805056 |

## 固定 validation 选择性曲线差值

| Seed | Commission area reduction | F1 area gain |
|---:|---:|---:|
| 42 | +0.023 pp | -0.034 pp |
| 43 | +0.087 pp | +0.035 pp |
| 44 | -0.004 pp | -0.048 pp |

- Mean Commission-area reduction：+0.035 pp
- Mean F1-area gain：-0.015 pp

## Gate

- parameter_count_matched: **True**
- commission_mean_not_worse: **True**
- f1_mean_not_worse: **False**
- commission_seed_wins: **True**
- f1_seed_wins: **False**
- meaningful_effect_passed: **False**
- passed: **False**

STOP：垂直 token 未证明对全局质量特征有互补增益。
