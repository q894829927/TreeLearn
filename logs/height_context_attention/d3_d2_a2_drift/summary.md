# D1 HSCA 跨域预测漂移诊断

- 本诊断不读取 GT，不选择 checkpoint，不修改 Wytham 参数。
- 诊断 suite：d2_a2
- 运行模式：full

| Domain | Model | Points | Tree→non-tree/base-tree | Seed removed/base-seed | |Δtree prob| | XY residual | Z residual | Gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| l1w | d2_a2_seed_identity_hsca | 6,105,000 | 3.6753% | 4.2287% | 0.036907 | 0.0819 m | 0.1863 m | 0.2396 |
| wytham | d2_a2_seed_identity_hsca | 10,045,000 | 2.2511% | 5.5517% | 0.024306 | 0.0817 m | 0.1944 m | 0.2373 |

## Wytham / L1W 漂移倍率

### d2_a2_seed_identity_hsca

| Metric | Ratio |
|---|---:|
| tree_to_non_tree_rate_among_base_tree | 0.613x |
| non_tree_to_tree_rate_among_base_non_tree | 1.761x |
| seed_removed_rate_among_base_seed | 1.313x |
| seed_added_rate_among_non_base_seed | 2.845x |
| seed_removed_semantic_rate_among_base_seed | 1.305x |
| seed_removed_offset_rate_among_base_seed | 1.313x |
| abs_probability_delta_mean | 0.659x |
| offset_xy_residual_mean | 0.998x |
| offset_z_residual_mean | 1.044x |
| gate_mean | 0.990x |

## 解释规则

- Tree→non-tree 与 seed removed 明显放大：优先处理 semantic/seed 保守化。
- Offset residual 明显放大而 semantic 稳定：优先处理 offset identity preservation。
- Gate 跨域升高并伴随残差放大：门控或高度归一化发生域偏移。
- A1/A2 都产生相似漂移：问题来自适配目标，而不是注意力本身。
