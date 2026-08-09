# E7b 域稳健比例过滤复核：Wytham

- 数据角色：**development_only**
- 锁定 keep ratio：0.8500
- 保留实例：1583 / 1862
- 质量评分耗时占完整 pipeline：5.485%

| Run | Completeness | Commission | F1 | Precision | Recall | Coverage |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 64.766249% | 18.741059% | 72.081218% | 62.458914% | 80.493762% | 57.718038% |
| filtered | 63.169897% | 12.893082% | 73.231989% | 58.733170% | 78.328076% | 55.769467% |

## Delta

- F1：+1.150771 个百分点
- Commission 降低：+5.847977 个百分点
- Completeness 下降：+1.596351 个百分点

## Gate

- checkpoint_seed_locked: **True**
- filter_enabled: **True**
- filter_mode_locked: **True**
- keep_ratio_locked: **True**
- kept_count_correct: **True**
- completeness_drop_passed: **False**
- effect_size_passed: **True**
- runtime_overhead_passed: **True**
- passed: **False**

STOP
