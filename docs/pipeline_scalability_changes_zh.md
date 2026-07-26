# 双锚点 Pipeline 规模控制与实验约定

本文记录 2026-07-26 针对全森林实例聚类耗时过长所做的修改。

## 核心实验约定

- A--E 的 `use_upper_seeds` 均为 `False`，因此使用相同的 base-seed
  候选集合。B--E 可以改变双锚点特征和树轴特征，但不会同时改变聚类样本。
- F 是独立的 upper-seed 消融，仅与 E 比较，不放入核心 A--E 逐项消融表。
- A 是 robust-base 单锚点基线。A0 使用原版 TreeLearn 的 legacy base
  anchor，应作为额外的原版参考单独报告。
- 已训练的 T2 使用 robust base，不需要因为本次 Pipeline 性能修改而重新训练。

## 新增配置

```yaml
grouping:
  use_upper_seeds: False
  max_cluster_seed_points: 500000
  knn_chunk_size: 200000
```

- `use_upper_seeds`：是否把 upper-anchor 附近点加入初始聚类种子。
- `max_cluster_seed_points`：超过该数量时停止并报错，防止误启动多小时的
  全局 HDBSCAN。
- `knn_chunk_size`：分块分配剩余点，降低 KNN 查询阶段的峰值内存。

重叠 tile 的 ensemble 已由 Pandas 宽表 `concat + groupby` 改为 NumPy：

1. 坐标圆整到厘米；
2. 用结构化坐标键生成唯一坐标和 inverse group id；
3. 每个预测数组逐列使用 `numpy.bincount` 累加并求均值。

这种实现保持原来的坐标分组和均值语义，但不会同时创建包含全部 backbone
通道的宽 DataFrame。日志会记录输入点数、唯一点数和 NumPy ensemble 耗时。

正式指标实验默认：

```yaml
save_cfg:
  save_pointwise: False
  save_treewise: False
```

无论这两个开关是否关闭，`full_forest/*.laz` 都会正常保存，可直接用于官方
evaluation。最终定性可视化时再临时开启点级和单树输出。

## 推荐运行

已有 T2 checkpoint 时先运行 E：

```bash
python tools/pipeline/pipeline.py \
  --config configs/experiments/pipeline_e_full.yaml
```

日志必须出现类似：

```text
Clustering 123,456 seed points in 5D
Clustering finished in 321.0s
Assigned 456,789 remaining points in 42.0s (chunk size: 200,000)
```

如果日志没有 seed 数量，说明服务器代码尚未同步本次修改。

upper-seed 扩展消融：

```bash
python tools/pipeline/pipeline.py \
  --config configs/experiments/pipeline_f_upper_seeds.yaml
```

若 F 超过 `max_cluster_seed_points`，不要直接无限增大阈值。应先记录候选点数、
内存和运行时间，再决定是否采用空间分块或种子下采样。

## 原版参考

需要额外报告原版 TreeLearn 时，训练：

```bash
python tools/training/train.py \
  --config configs/experiments/train_t0_original.yaml \
  --work_dir t0_original
```

随后运行 `pipeline_a0_original.yaml` 和 `evaluate_a0_original.yaml`。

核心 A--E 仍使用 robust base，避免把 base-anchor 变化混入双锚点逐项消融。
