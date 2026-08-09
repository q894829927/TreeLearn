# E10 Wytham 锁定随机排序置换检验

- Wytham 不参与模型、比例网格或 Gate 参数选择。
- 随机排序置换：10000 次。

| Metric | Vertical-MLP observed | Random null mean | Random 95% interval | One-sided p |
|---|---:|---:|---:|---:|
| Commission area（越低越好） | 10.989% | 18.741% | [17.346%, 20.117%] | 0.000100 |
| Detection F1 area（越高越好） | 70.562% | 60.259% | [58.835%, 61.670%] | 0.000100 |

## Gate

- score_only_metadata_passed: **True**
- baseline_counts_reproduced: **True**
- commission_permutation_passed: **True**
- f1_permutation_passed: **True**
- passed: **True**
