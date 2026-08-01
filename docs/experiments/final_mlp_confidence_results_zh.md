# 冻结主干的置信度引导 Seed 筛选：最终实验记录

## 1. 最终锁定方法

最终方法不使用 Point Transformer，也不融合预测的残差向量：

```text
冻结的官方 TreeLearn 主干与原始 heads
        ↓
逐点 backbone feature
        ↓
轻量 MLP：base-residual XY + 异方差置信度
        ↓
只按照置信度保留排名前 78% 的原始 base seeds
        ↓
原版二维 HDBSCAN 聚类与剩余点分配
```

锁定参数：

- `axis_branch_type: mlp`
- `axis_target_mode: base_residual`
- `use_axis_fusion: False`
- `seed_confidence_filter_mode: top_ratio`
- `seed_confidence_keep_ratio: 0.78`
- TreeLearn 主干及原始 heads 永久冻结
- 最终 checkpoint 按 323 个验证 tiles 的残差 Mean Error 选择，不按 L1W/Wytham 指标选择

正式 checkpoint：

```text
work_dirs/base_residual_mlp_frozen_s43/best_base_residual_xy.pth
```

选择原因：seed 43 的验证 Mean Error 为 0.326 m，在 seed 42/43/44 中最低。

## 2. L1W 主要结果

| 方法 | Completeness | Commission | F1 | Precision | Recall | Coverage |
|---|---:|---:|---:|---:|---:|---:|
| Base-only / r100 | 100.0% | 3.1% | 98.4% | 98.9% | 99.1% | 98.0% |
| Random-78% s42 | 100.0% | 2.5% | 98.7% | 98.9% | 99.1% | 98.0% |
| Random-78% s43 | 100.0% | 1.9% | 99.0% | 98.9% | 99.1% | 98.0% |
| Random-78% s44 | 100.0% | 2.5% | 98.7% | 98.9% | 99.1% | 98.0% |
| MLP confidence-78% s42 | 99.4% | 0.0% | 99.7% | 98.5% | 99.2% | 97.7% |
| MLP confidence-78% s43 | 100.0% | 0.6% | 99.7% | 98.9% | 99.1% | 98.0% |
| MLP confidence-78% s44 | 100.0% | 0.0% | 100.0% | 98.8% | 99.2% | 98.0% |
| Point Transformer confidence-78% | 97.4% | 0.7% | 98.4% | 96.3% | 99.1% | 95.5% |

汇总：

- Random-78% 三次 F1 平均约 98.8%，Commission 平均约 2.3%；
- MLP confidence-78% 三次 F1 平均约 99.8%，Commission 平均约 0.2%；
- learned confidence 相对随机等量筛选的 F1 优势约 1.0 个百分点；
- Point Transformer 明显低于 MLP，注意力路线淘汰。

## 3. Wytham 跨域结果

| 方法 | Completeness | Commission | F1 | Precision | Recall | Coverage |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 64.8% | 18.7% | 72.1% | 62.5% | 80.5% | 57.7% |
| MLP confidence-78% s42 | 64.5% | 12.5% | 74.3% | 60.0% | 81.1% | 56.7% |
| MLP confidence-78% s43 locked | 64.5% | 13.5% | 73.9% | 60.3% | 80.9% | 56.8% |

seed 43 锁定结果相对 baseline：

- F1：提高 1.8 个百分点；
- Commission：降低 5.2 个百分点；
- Completeness：降低 0.3 个百分点；
- Precision：降低 2.2 个百分点；
- Recall：提高 0.4 个百分点；
- Coverage：降低 0.9 个百分点。

因此论文主结论应写为：置信度引导的 seed 筛选可以减少复杂森林中的假阳性实例并提高检测 F1。不能写成分割 Precision、Coverage 或整体分割质量全面提升。

## 4. 实验有效性与限制

1. L1W 被用于选择 Top-78% 比例，因此属于开发/验证 benchmark。
2. Wytham 在早期绝对阈值实验中已经被观察，后续 r078 应描述为跨域确认，而不是完全未见的最终测试。
3. 若投稿需要严格的独立测试结论，应增加第三个未用于模型或超参数选择的带真值森林数据集。
4. 不再根据 Wytham 结果修改 keep ratio、置信度、模型或聚类参数。
5. Point Transformer 没有带来下游提升，不能作为论文有效模块或标题贡献。

## 5. 论文最小消融

论文正文保留以下四组即可：

1. Base-only：证明原始聚类性能；
2. Random-78%：排除仅因减少 seeds 获益；
3. MLP confidence-78%：最终方法；
4. Point Transformer confidence-78%：说明更复杂注意力并非收益来源，可放附录。

建议正文同时报告 Detection F1、Completeness、Commission，以及 Segmentation Precision、Recall、Coverage，避免只选择有利指标。

## 6. 后续工作

模型实验停止。接下来只进行：

1. 从 `evaluation_results.pt` 提取未四舍五入的精确指标；
2. 统计参数量、运行时间和峰值显存/内存；
3. 生成 baseline 与最终方法的定性对比图，重点展示假阳性树减少的区域；
4. 准备第三个未见带真值数据集，或在论文中明确 Wytham 的开发使用限制；
5. 撰写方法、实验设置、消融和局限性章节。

## 7. 精确指标汇总命令

`evaluation_results.pt` 中原有的汇总百分比已经被评估程序四舍五入到 0.1%。使用以下脚本从匹配数量和逐树结果重新计算未四舍五入指标：

```bash
python tools/diagnostics/summarize_final_results.py \
  --output_json logs/final_exact_metrics.json \
  --output_md logs/final_exact_metrics.md \
  2>&1 | tee logs/summarize_final_results.log
```

正常情况下必须找到 11 组结果并生成：

```text
logs/final_exact_metrics.json
logs/final_exact_metrics.md
```

脚本同时计算 L1W Random-78% 和 MLP confidence-78% 三个 seed 的均值与样本标准差（`ddof=1`）。如果报告缺少结果文件，先检查对应 pipeline/evaluation 输出目录，不要使用 `--allow_missing` 生成正式论文表格。
