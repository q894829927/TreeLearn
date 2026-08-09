# E0 复杂度感知种子注意力：Oracle 诊断

- Pipeline config：`configs/experiments/point_transformer/pipeline_l1w_seed_ratio_r100.yaml`
- 点数：20,431,825
- 基础候选种子：325,596
- 固定保留率：0.780

## 聚类检测结果

| Mode | TP | FP | FN | Completeness | Commission | F1 |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 156 | 5 | 0 | 100.000% | 3.106% | 98.423% |
| oracle_global | 143 | 4 | 13 | 91.667% | 2.721% | 94.389% |
| oracle_balanced | 143 | 4 | 13 | 91.667% | 2.721% | 94.389% |
| random_s42 | 156 | 4 | 0 | 100.000% | 2.500% | 98.734% |
| random_s43 | 156 | 3 | 0 | 100.000% | 1.887% | 99.048% |
| random_s44 | 156 | 4 | 0 | 100.000% | 2.500% | 98.734% |

## Oracle 相对收益

- F1 gain：-4.033 pp
- Commission reduction：+0.385 pp
- Completeness drop：+8.333 pp
- F1 gain over random mean：-4.449 pp

## 种子质量

| Mode | Selected | Tree coverage | ≥tau_min coverage | Mean vote error | P90 vote error | Mean utility |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 325,596 | 100.000% | 100.000% | 0.3278 m | 0.5627 m | 0.3909 |
| oracle_global | 253,965 | 100.000% | 100.000% | 0.3094 m | 0.5417 m | 0.5012 |
| oracle_balanced | 253,965 | 100.000% | 100.000% | 0.3094 m | 0.5417 m | 0.5012 |

## Gate

- minimum_f1_gain_pp: **False**
- minimum_commission_reduction_pp: **False**
- maximum_completeness_drop_pp: **False**
- minimum_gain_over_random_pp: **False**
- passed: **False**

STOP：种子选择上限不足，不训练复杂度感知种子注意力网络。
