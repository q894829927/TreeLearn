# Q2 参数匹配 Coverage-CVaR 实例质量 MLP

- 数据：固定 train/validation forests；未读取 Wytham。
- 固定拒绝比例：15.0%
- 随机种子：[42, 43, 44]
- 三个模型使用完全相同的全局特征、encoder 与参数量。

| Model | Safe AP | Critical AP | F1 | Completeness | Commission | F1 gain vs unfiltered |
|---|---:|---:|---:|---:|---:|---:|
| quality_iou_control | 0.987410 | 0.994345 | 86.763% | 99.421% | 23.036% | +7.461 pp |
| safe_iou_control | 0.987108 | 0.993652 | 86.760% | 99.401% | 23.028% | +7.459 pp |
| coverage_cvar | 0.986902 | 0.993028 | 86.706% | 99.339% | 23.076% | +7.404 pp |

## Coverage-CVaR 相对 safe+IoU 参数匹配控制组

- F1：-0.054 pp
- Completeness：-0.062 pp
- Commission reduction：-0.048 pp
- Seed wins：0 / 3
- Validation plot wins：1 / 5

## 每森林 F1 差值

| Plot | Coverage-CVaR minus safe+IoU |
|---|---:|
| G4N | -0.403 pp |
| G4W | +0.031 pp |
| L1N | +0.000 pp |
| O1N | +0.000 pp |
| O1W | +0.000 pp |

## Gate

- q1_gate_passed: **True**
- expected_seeds_present: **True**
- parameter_count_matched: **True**
- coverage_completeness_preserved: **True**
- coverage_f1_gain_over_unfiltered_passed: **True**
- coverage_commission_reduction_passed: **True**
- coverage_gain_over_control_passed: **False**
- coverage_completeness_vs_control_passed: **True**
- seed_consistency_passed: **False**
- plot_consistency_passed: **False**
- coverage_seed_stability_passed: **True**
- passed: **False**

STOP：Coverage-CVaR 未优于参数匹配 safe+IoU 控制组，不实现关系注意力，不使用 Wytham 调参。
