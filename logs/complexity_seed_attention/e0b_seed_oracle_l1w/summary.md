# E0b 未标注保护的空间约束 Seed Oracle

- 候选种子：325,596
- 固定保留数：253,965
- 未标注保护种子：151,361
- 空间/树级保底种子：8,147
- Masked/Spatial Jaccard：0.992922

## 检测结果

| Mode | TP | FP | FN | Completeness | Commission | F1 |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 156 | 5 | 0 | 100.000% | 3.106% | 98.423% |
| oracle_masked | 137 | 1 | 19 | 87.821% | 0.725% | 93.197% |
| oracle_spatial | 156 | 0 | 0 | 100.000% | 0.000% | 100.000% |
| random_spatial_s42 | 156 | 3 | 0 | 100.000% | 1.887% | 99.048% |
| random_spatial_s43 | 156 | 4 | 0 | 100.000% | 2.500% | 98.734% |
| random_spatial_s44 | 156 | 2 | 0 | 100.000% | 1.266% | 99.363% |

## 选择集合诊断

| Mode | Retained | Tree | Non-tree | Unknown | Cell coverage | Min tree count | Mean error |
|---|---:|---:|---:|---:|---:|---:|---:|
| oracle_masked | 253,965 | 102,604 | 0 | 151,361 | 55.044% | 0 | 0.3038 m |
| oracle_spatial | 253,965 | 102,604 | 0 | 151,361 | 100.000% | 50 | 0.3049 m |
| random_spatial_s42 | 253,965 | 100,505 | 2,099 | 151,361 | 100.000% | 184 | 0.3490 m |
| random_spatial_s43 | 253,965 | 100,461 | 2,143 | 151,361 | 100.000% | 177 | 0.3491 m |
| random_spatial_s44 | 253,965 | 100,496 | 2,108 | 151,361 | 100.000% | 181 | 0.3490 m |

## 效果

- f1_gain_pp: +1.577287
- commission_reduction_pp: +3.105590
- completeness_drop_pp: +0.000000
- f1_gain_over_random_mean_pp: +0.951715
- random_mean_f1: +0.990483

## Gate

- minimum_f1_gain_pp: **True**
- minimum_commission_reduction_pp: **True**
- maximum_completeness_drop_pp: **True**
- minimum_gain_over_random_pp: **True**
- passed: **True**

PASS：进入冻结主干的 Seed-MLP 控制实验。
