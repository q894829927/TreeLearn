# Q1 覆盖保护型实例质量 Oracle

- 数据：固定 13 个 train forests 与 5 个 validation forests。
- 未读取 Wytham；未训练网络；未修改实例预测。
- 固定最大拒绝比例：15.0%

## 标签定义

- `coverage_critical`：每棵已匹配 GT 树中 IoU 最高的唯一代表实例。
- `safe_reject`：classification-valid 假阳性或冗余正实例。
- `protected_unknown`：边界、低标注覆盖或 IoU 模糊实例，Oracle 不删除。

## Split 汇总

| Split | Instances | Safe reject | Coverage critical | Unknown | Baseline F1 | Oracle F1 | F1 gain | Commission reduction | Completeness drop |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 6,807 | 1,957 | 4,566 | 284 | 82.352% | 90.300% | +7.948 pp | +12.316 pp | +0.000 pp |
| validation | 2,595 | 842 | 1,613 | 140 | 79.302% | 87.307% | +8.005 pp | +11.771 pp | +0.000 pp |

## Validation 每森林

| Plot | Instances | Safe reject | Critical | F1 gain | Commission reduction | Completeness drop |
|---|---:|---:|---:|---:|---:|---:|
| G4N | 364 | 63 | 281 | +8.504 pp | +15.211 pp | +0.000 pp |
| G4W | 653 | 84 | 537 | +7.254 pp | +13.527 pp | +0.000 pp |
| L1N | 685 | 235 | 429 | +8.080 pp | +11.726 pp | +0.000 pp |
| O1N | 190 | 33 | 150 | +8.271 pp | +14.807 pp | +0.000 pp |
| O1W | 703 | 427 | 216 | +7.003 pp | +6.556 pp | +0.000 pp |

## Gate

- expected_train_plots: **True**
- expected_validation_plots: **True**
- enough_train_safe_reject: **True**
- enough_train_coverage_critical: **True**
- enough_validation_safe_reject: **True**
- enough_validation_coverage_critical: **True**
- oracle_f1_gain_passed: **True**
- oracle_commission_reduction_passed: **True**
- coverage_preserved: **True**
- per_plot_consistency: **True**
- labels_mutually_exclusive: **True**
- passed: **True**

PASS：进入 Q2 双头 MLP 控制实验；先验证覆盖保护目标，不实现注意力。
