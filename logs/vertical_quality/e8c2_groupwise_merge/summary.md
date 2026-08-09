# E8c2 Groupwise 安全合并头

- 训练源实例：993
- 验证源实例：370
- 验证正目标源实例：258
- 锁定部署 seed：42
- 锁定阈值：0.9324288964271545
- 可训练参数：4354
- Pointwise 安全召回：0.016796
- Groupwise 安全召回增益：-0.005168

| Metric | Mean | Std | Min | Max |
|---|---:|---:|---:|---:|
| source_average_precision | 0.642634 | 0.005392 | 0.638790 | 0.648798 |
| top1_correct_rate_on_positive_sources | 0.645995 | 0.004476 | 0.643411 | 0.651163 |
| pair_precision | 1.000000 | 0.000000 | 1.000000 | 1.000000 |
| positive_source_recall | 0.011628 | 0.003876 | 0.007752 | 0.015504 |
| unsafe_merge_rate | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

- Gate passed seeds：0/3

## Gate

- pointwise_e8c_failed_as_expected: **True**
- enough_seed_passes: **False**
- locked_seed_passed: **False**
- recall_stability_passed: **True**
- recall_gain_over_pointwise_passed: **False**
- passed: **False**

STOP：不进入 pipeline 合并。
