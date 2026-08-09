# E8a 质量引导合并 Oracle：wytham_development_upper_bound

- 数据角色：**development_only**
- 锁定低质量比例：0.1500
- 低质量实例：279 / 1862
- Oracle 可合并实例：44
- 未找到安全目标并保留：235

| Run | TP | FP | FN | Completeness | Commission | F1 |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 568 | 131 | 309 | 64.766249% | 18.741059% | 72.081218% |
| merge_oracle | 570 | 112 | 307 | 64.994299% | 16.422287% | 73.123797% |

## 低质量实例构成

- TP：14
- counted FP：49
- ignored unmatched：216

## Oracle 合并来源构成

- TP：0
- counted FP：15
- ignored unmatched：29

## Gate

- minimum_f1_gain_pp: **0.5**
- minimum_commission_reduction_pp: **2.0**
- maximum_completeness_drop_pp: **0.5**
- f1_gain_pp: **1.0425790318536912**
- commission_reduction_pp: **2.318771265192421**
- completeness_drop_pp: **-0.22805017103762282**
- f1_gain_passed: **True**
- commission_reduction_passed: **True**
- completeness_passed: **True**
- passed: **True**

PASS：进入 E8b validation 邻接对数据生成与无标签合并规则。
