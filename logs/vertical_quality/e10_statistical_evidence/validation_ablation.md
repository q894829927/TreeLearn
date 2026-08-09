# E10 Validation 配对消融：Vertical-MLP vs Global-MLP

- Bootstrap：5000 次；同时重采样森林和训练 seed。
- 所有重采样均为配对重采样，两个模型使用相同森林和 seed。

| Metric | Global-MLP | Vertical-MLP | Paired delta | 95% CI |
|---|---:|---:|---:|---:|
| Commission area | 18.783% | 18.888% | -0.105 pp reduction | [-0.459, +0.315] |
| Detection F1 area | 85.202% | 85.027% | -0.175 pp | [-0.451, +0.137] |

## 每个训练 seed 的配对差值

| Seed | Commission area reduction | F1 area gain |
|---:|---:|---:|
| 42 | -0.035 pp | -0.099 pp |
| 43 | -0.197 pp | -0.255 pp |
| 44 | -0.081 pp | -0.172 pp |

## Gate

- commission_mean_improved: **False**
- f1_mean_improved: **False**
- commission_seed_wins: **False**
- f1_seed_wins: **False**
- passed: **False**
