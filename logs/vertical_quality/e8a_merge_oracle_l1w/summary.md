# E8a 质量引导合并 Oracle：l1w_sanity

- 数据角色：**regression_validation**
- 锁定低质量比例：0.1500
- 低质量实例：55 / 371
- Oracle 可合并实例：10
- 未找到安全目标并保留：45

| Run | TP | FP | FN | Completeness | Commission | F1 |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 156 | 5 | 0 | 100.000000% | 3.105590% | 98.422713% |
| merge_oracle | 156 | 1 | 0 | 100.000000% | 0.636943% | 99.680511% |

## 低质量实例构成

- TP：0
- counted FP：4
- ignored unmatched：51

## Oracle 合并来源构成

- TP：0
- counted FP：4
- ignored unmatched：6

## Gate

- minimum_f1_gain_pp: **0.5**
- minimum_commission_reduction_pp: **2.0**
- maximum_completeness_drop_pp: **0.5**
- f1_gain_pp: **1.2577982483546748**
- commission_reduction_pp: **2.468647386952566**
- completeness_drop_pp: **0.0**
- f1_gain_passed: **True**
- commission_reduction_passed: **True**
- completeness_passed: **True**
- passed: **True**

PASS：进入 E8b validation 邻接对数据生成与无标签合并规则。
