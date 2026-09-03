# D1 HSCA 跨域预测漂移诊断

- 本诊断不读取 GT，不选择 checkpoint，不修改 Wytham 参数。
- 诊断 suite：d2_a2
- 运行模式：smoke

| Domain | Model | Points | Tree→non-tree/base-tree | Seed removed/base-seed | |Δtree prob| | XY residual | Z residual | Gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| l1w | d2_a2_seed_identity_hsca | 10,000 | 6.6943% | 3.2727% | 0.068832 | 0.0766 m | 0.1991 m | 0.2278 |
| wytham | d2_a2_seed_identity_hsca | 10,000 | 4.2069% | 12.9630% | 0.044340 | 0.0786 m | 0.1752 m | 0.2375 |

## Wytham / L1W 漂移倍率

### d2_a2_seed_identity_hsca

| Metric | Ratio |
|---|---:|
| tree_to_non_tree_rate_among_base_tree | 0.628x |
| non_tree_to_tree_rate_among_base_non_tree | 0.666x |
| seed_removed_rate_among_base_seed | 3.961x |
| seed_added_rate_among_non_base_seed | 0.989x |
| seed_removed_semantic_rate_among_base_seed | N/A |
| seed_removed_offset_rate_among_base_seed | 3.961x |
| abs_probability_delta_mean | 0.644x |
| offset_xy_residual_mean | 1.027x |
| offset_z_residual_mean | 0.880x |
| gate_mean | 1.043x |

## 解释规则

- Tree→non-tree 与 seed removed 明显放大：优先处理 semantic/seed 保守化。
- Offset residual 明显放大而 semantic 稳定：优先处理 offset identity preservation。
- Gate 跨域升高并伴随残差放大：门控或高度归一化发生域偏移。
- A1/A2 都产生相似漂移：问题来自适配目标，而不是注意力本身。
