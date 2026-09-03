# D1 HSCA 跨域预测漂移诊断

- 本诊断不读取 GT，不选择 checkpoint，不修改 Wytham 参数。
- 运行模式：full

| Domain | Model | Points | Tree→non-tree/base-tree | Seed removed/base-seed | |Δtree prob| | XY residual | Z residual | Gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| l1w | a1_height_mlp | 6,105,000 | 3.1525% | 9.6879% | 0.033821 | 0.0596 m | 0.2689 m | 0.1587 |
| l1w | a2_hsca | 6,105,000 | 4.4602% | 8.9868% | 0.040759 | 0.0687 m | 0.2447 m | 0.3131 |
| wytham | a1_height_mlp | 10,045,000 | 3.7063% | 18.6790% | 0.035906 | 0.0600 m | 0.2768 m | 0.1653 |
| wytham | a2_hsca | 10,045,000 | 3.4649% | 15.3788% | 0.031886 | 0.0677 m | 0.2505 m | 0.3137 |

## Wytham / L1W 漂移倍率

### a1_height_mlp

| Metric | Ratio |
|---|---:|
| tree_to_non_tree_rate_among_base_tree | 1.176x |
| seed_removed_rate_among_base_seed | 1.928x |
| seed_removed_semantic_rate_among_base_seed | 2.916x |
| seed_removed_offset_rate_among_base_seed | 0.882x |
| abs_probability_delta_mean | 1.062x |
| offset_xy_residual_mean | 1.005x |
| offset_z_residual_mean | 1.029x |
| gate_mean | 1.041x |

### a2_hsca

| Metric | Ratio |
|---|---:|
| tree_to_non_tree_rate_among_base_tree | 0.777x |
| seed_removed_rate_among_base_seed | 1.711x |
| seed_removed_semantic_rate_among_base_seed | 2.831x |
| seed_removed_offset_rate_among_base_seed | 1.144x |
| abs_probability_delta_mean | 0.782x |
| offset_xy_residual_mean | 0.986x |
| offset_z_residual_mean | 1.024x |
| gate_mean | 1.002x |

## 解释规则

- Tree→non-tree 与 seed removed 明显放大：优先处理 semantic/seed 保守化。
- Offset residual 明显放大而 semantic 稳定：优先处理 offset identity preservation。
- Gate 跨域升高并伴随残差放大：门控或高度归一化发生域偏移。
- A1/A2 都产生相似漂移：问题来自适配目标，而不是注意力本身。
