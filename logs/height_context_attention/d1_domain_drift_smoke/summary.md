# D1 HSCA 跨域预测漂移诊断

- 本诊断不读取 GT，不选择 checkpoint，不修改 Wytham 参数。
- 运行模式：smoke

| Domain | Model | Points | Tree→non-tree/base-tree | Seed removed/base-seed | |Δtree prob| | XY residual | Z residual | Gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| l1w | a1_height_mlp | 10,000 | 5.6689% | 6.9091% | 0.062130 | 0.0562 m | 0.3213 m | 0.1563 |
| l1w | a2_hsca | 10,000 | 6.9593% | 8.0000% | 0.069435 | 0.0672 m | 0.2653 m | 0.3014 |
| wytham | a1_height_mlp | 10,000 | 5.2725% | 19.1358% | 0.056333 | 0.0597 m | 0.2979 m | 0.1750 |
| wytham | a2_hsca | 10,000 | 5.1948% | 20.3704% | 0.051326 | 0.0656 m | 0.2441 m | 0.3158 |

## Wytham / L1W 漂移倍率

### a1_height_mlp

| Metric | Ratio |
|---|---:|
| tree_to_non_tree_rate_among_base_tree | 0.930x |
| seed_removed_rate_among_base_seed | 2.770x |
| seed_removed_semantic_rate_among_base_seed | 2.160x |
| seed_removed_offset_rate_among_base_seed | 3.607x |
| abs_probability_delta_mean | 0.907x |
| offset_xy_residual_mean | 1.063x |
| offset_z_residual_mean | 0.927x |
| gate_mean | 1.120x |

### a2_hsca

| Metric | Ratio |
|---|---:|
| tree_to_non_tree_rate_among_base_tree | 0.746x |
| seed_removed_rate_among_base_seed | 2.546x |
| seed_removed_semantic_rate_among_base_seed | 2.037x |
| seed_removed_offset_rate_among_base_seed | 2.696x |
| abs_probability_delta_mean | 0.739x |
| offset_xy_residual_mean | 0.975x |
| offset_z_residual_mean | 0.920x |
| gate_mean | 1.048x |

## 解释规则

- Tree→non-tree 与 seed removed 明显放大：优先处理 semantic/seed 保守化。
- Offset residual 明显放大而 semantic 稳定：优先处理 offset identity preservation。
- Gate 跨域升高并伴随残差放大：门控或高度归一化发生域偏移。
- A1/A2 都产生相似漂移：问题来自适配目标，而不是注意力本身。
