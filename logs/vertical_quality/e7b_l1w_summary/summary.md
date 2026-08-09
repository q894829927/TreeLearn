# E7b 域稳健比例过滤复核：L1W

- 数据角色：**regression_validation**
- 锁定 keep ratio：0.8500
- 保留实例：316 / 371
- 质量评分耗时占完整 pipeline：3.776%

| Run | Completeness | Commission | F1 | Precision | Recall | Coverage |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 100.000000% | 3.105590% | 98.422713% | 98.914754% | 99.069687% | 98.022950% |
| filtered | 100.000000% | 0.636943% | 99.680511% | 98.914965% | 99.069458% | 98.022933% |

## Delta

- F1：+1.257798 个百分点
- Commission 降低：+2.468647 个百分点
- Completeness 下降：+0.000000 个百分点

## Gate

- checkpoint_seed_locked: **True**
- filter_enabled: **True**
- filter_mode_locked: **True**
- keep_ratio_locked: **True**
- kept_count_correct: **True**
- completeness_drop_passed: **True**
- effect_size_passed: **True**
- runtime_overhead_passed: **True**
- passed: **True**

PASS
