# 冻结 TreeLearn 主干的 Point Transformer Axis-XY 实验

## 1. 实验目的与固定协议

本实验用于验证：在不修改官方 TreeLearn 表征能力的前提下，轻量级局部
Point Transformer 是否能够有效补充树轴形态信息。

下列原始模块全部冻结，其 BatchNorm 层始终保持评估模式：

```text
input_conv、unet、output_layer、semantic_linear、offset_linear
```

可训练分支结构如下：

```text
32维逐点主干特征
  + verticality
  + 根据预测基部计算的归一化离地高度
  -> 输入投影
  -> 可选的单层局部 Point Transformer
  -> Axis-XY 和 log variance
```

Point Transformer 只处理语义分支预测为树的候选点。候选点首先在
`0.4 m` 体素网格中进行平均池化，每个查询点在相邻的 `3×3×3` 个体素中
选择最近的 8 个已占用体素作为局部邻居。该实现完全基于 PyTorch，不需要
安装 `torch-cluster` 或编译额外 CUDA 扩展。

为兼容 24GB 显存设备，Point Transformer 训练配置默认使用
`batch_size: 1`，并将 query 按 8192 个点分块，通过梯度检查点重算注意力
中间结果。query 分块只改变峰值显存和运行时间，不改变邻域或预测公式。

监督目标和损失定义如下：

```text
Axis-XY target = upper_offset[:2] - base_offset[:2]
有效掩码       = masks_off & masks_upper & predicted_tree_mask
逐点误差       = SmoothL1(predicted Axis-XY, target Axis-XY)
损失           = exp(-log_variance) * point_error + log_variance
置信度         = sigmoid(-log_variance)
```

本轮实验继续沿用树高 55%～75% 范围内的 upper anchor，不同时修改标签定义。
Wytham 作为最终外部测试集，不得用于选择 checkpoint 或融合权重。

## 2. 环境与运行前检查

在 TreeLearn 仓库根目录运行：

```bash
conda activate TreeLearn
mkdir -p logs
git branch --show-current
python -m unittest discover -s tests -p 'test_axis_point_transformer.py' -v
```

当前分支应为：

```text
point-transformer
```

训练前检查关键配置能否正常加载：

```bash
python - <<'PY'
from tree_learn.util import get_config

files = [
    "configs/experiments/point_transformer/train_axis_mlp_frozen.yaml",
    "configs/experiments/point_transformer/train_axis_pt_frozen.yaml",
    "configs/experiments/point_transformer/pipeline_l1w_axis_pt_w000.yaml",
    "configs/experiments/point_transformer/pipeline_wytham_axis_pt_final.yaml",
]

for path in files:
    cfg = get_config(path)
    print(path, "OK")
    if hasattr(cfg, "model"):
        print("  branch:", getattr(cfg.model, "axis_branch_type", None))
        print("  use_axis:", getattr(cfg.model, "use_axis_branch", False))
PY
```

确认官方 small-tree checkpoint 存在：

```bash
test -f data/model_weights/model_weights_with_small_20241213.pth
```

如果该命令返回非零状态，必须先补齐 checkpoint，不能直接开始训练。

## 3. 阶段 A：逐点 MLP 控制组

MLP 控制组不使用局部注意力，仅使用相同的基础特征预测 Axis-XY 和
Confidence。它用于判断后续性能提升是否确实来自 Point Transformer。

启动训练：

```bash
nohup python -u tools/training/train.py \
  --config configs/experiments/point_transformer/train_axis_mlp_frozen.yaml \
  --work_dir axis_mlp_frozen \
  > logs/train_axis_mlp_frozen.log 2>&1 < /dev/null &

echo $! | tee logs/train_axis_mlp_frozen.pid
tail -f logs/train_axis_mlp_frozen.log
```

验证阶段每当平均 Axis-XY 误差下降时，程序会自动保存：

```text
work_dirs/axis_mlp_frozen/best_axis_xy.pth
```

训练结束后，检查所有官方共享参数是否保持逐位不变：

```bash
python tools/diagnostics/verify_axis_checkpoint.py \
  --reference data/model_weights/model_weights_with_small_20241213.pth \
  --candidate work_dirs/axis_mlp_frozen/best_axis_xy.pth
```

只有输出以下信息才能继续：

```text
PASS: original TreeLearn tensors are bitwise unchanged.
```

## 4. 阶段 B：单层 Point Transformer

阶段 A 完成后再启动 Point Transformer 实验：

```bash
nohup python -u tools/training/train.py \
  --config configs/experiments/point_transformer/train_axis_pt_frozen.yaml \
  --work_dir axis_pt_frozen \
  > logs/train_axis_pt_frozen.log 2>&1 < /dev/null &

echo $! | tee logs/train_axis_pt_frozen.pid
tail -f logs/train_axis_pt_frozen.log
```

检查训练进程、显存和最新日志：

```bash
pgrep -af "[t]ools/training/train.py"
nvidia-smi
tail -n 50 logs/train_axis_pt_frozen.log
```

训练完成后检查冻结参数：

```bash
python tools/diagnostics/verify_axis_checkpoint.py \
  --reference data/model_weights/model_weights_with_small_20241213.pth \
  --candidate work_dirs/axis_pt_frozen/best_axis_xy.pth
```

运行单 tile 的预测、聚合、聚类和保存 smoke test：

```bash
python tools/diagnostics/smoke_axis_pipeline.py \
  --config configs/experiments/point_transformer/pipeline_l1w_axis_pt_w000.yaml \
  --output logs/axis_pipeline_smoke.npz
```

提取两组实验的验证指标：

```bash
grep "axis_xy_mean_error" logs/train_axis_mlp_frozen.log
grep "axis_xy_mean_error" logs/train_axis_pt_frozen.log
```

主要比较：

- `axis_xy_mean_error`
- `axis_xy_median_error`
- `axis_xy_p90_error`
- `axis_confidence_error_corr`

只有同时满足以下条件，才继续进行完整实例分割：

1. Point Transformer 的平均 Axis-XY 误差比 MLP 至少降低 5%；
2. Point Transformer 的 P90 误差不高于 MLP。

如果没有达到条件，应停止外部测试，优先重新设计 upper-anchor 标签，而不是
继续增加更多注意力层。

## 5. 阶段 C：在 L1W 上选择融合权重

置信度门控的二维投票为：

```text
base_vote_xy + weight * confidence * Axis-XY
```

固定测试以下四个权重：

```text
0、0.1、0.25、0.5
```

四组实验必须使用同一个 Point Transformer checkpoint。权重为 0 的实验必须
严格退化为 base-only 二维聚类，用于确认 pipeline 兼容性。

依次运行 pipeline 和评估：

```bash
for weight in 000 010 025 050; do
  nohup python -u tools/pipeline/pipeline.py \
    --config configs/experiments/point_transformer/pipeline_l1w_axis_pt_w${weight}.yaml \
    > logs/pipeline_l1w_axis_pt_w${weight}.log 2>&1 < /dev/null &

  pid=$!
  echo "$pid" | tee logs/pipeline_l1w_axis_pt_w${weight}.pid
  wait "$pid" || exit 1

  python -u tools/evaluation/evaluate.py \
    --config configs/experiments/point_transformer/evaluate_l1w_axis_pt_w${weight}.yaml \
    2>&1 | tee logs/evaluate_l1w_axis_pt_w${weight}.log
done
```

这些配置默认：

```yaml
tile_generation: False
```

因为可以复用已有的 L1W tiles。如果 `data/pipeline/L1W/tiles/npz` 不存在，
只将第一次 `w000` 实验的 `tile_generation` 改为 `True`。生成完成后立即恢复
为 `False`，后面三组实验必须复用完全相同的 tiles。

融合权重选择规则：

1. 优先选择 detection F1 最高的权重；
2. F1 相同时选择 Coverage 更高的权重；
3. F1 和 Coverage 都相同时选择更小的权重。

选定权重后先写入实验记录，再运行 Wytham。不能根据 Wytham 结果返回修改
融合权重。

## 6. 阶段 D：Wytham 最终测试

在第一次运行 Wytham 之前，将 L1W 选出的唯一融合权重写入：

```text
configs/experiments/point_transformer/pipeline_wytham_axis_pt_final.yaml
```

只允许修改：

```yaml
grouping:
  axis_fusion_weight: 选定的L1W权重
```

不得修改其他 Wytham 参数。

启动最终 pipeline：

```bash
nohup python -u tools/pipeline/pipeline.py \
  --config configs/experiments/point_transformer/pipeline_wytham_axis_pt_final.yaml \
  > logs/pipeline_wytham_axis_pt_final.log 2>&1 < /dev/null &

echo $! | tee logs/pipeline_wytham_axis_pt_final.pid
tail -f logs/pipeline_wytham_axis_pt_final.log
```

pipeline 完成后执行评估：

```bash
python -u tools/evaluation/evaluate.py \
  --config configs/experiments/point_transformer/evaluate_wytham_axis_pt_final.yaml \
  2>&1 | tee logs/evaluate_wytham_axis_pt_final.log
```

当前 Wytham 官方参考结果：

```text
Detection F1：72.0%
Coverage：    57.7%
Precision：   62.5%
Recall：      80.5%
```

只有满足以下任一条件，才继续进行论文消融：

- Detection F1 至少提升 0.5 个百分点；
- Coverage 至少提升 1.0 个百分点，同时 Precision 下降不超过 1.0 个百分点。

如果两个条件都不满足，应停止增加注意力模块，下一步改为重新设计 upper-anchor
标签。

## 7. 达标后进行最小消融

只有 Wytham 的 go/no-go 测试成功后，才运行以下四组消融：

1. 官方 base-only；
2. `pipeline_l1w_axis_mlp_w025.yaml`；
3. 选定融合权重的 Point Transformer；
4. `pipeline_l1w_axis_pt_no_conf.yaml`。

MLP 和关闭 Confidence 的评估 YAML 位于同一配置目录。

如果 L1W 最终选出的权重不是 `0.25`，在运行消融前，需要将 MLP 和
no-confidence 配置中的 `axis_fusion_weight` 同步改为已经锁定的权重。

## 8. 实验结果记录

训练完成后先填写以下信息，再分析结果：

| 项目 | MLP | Point Transformer |
|---|---:|---:|
| Git commit | | |
| Checkpoint 路径 | | |
| 最佳 epoch | | |
| 验证集 Axis-XY 平均误差（m） | | |
| 验证集 Axis-XY 中位误差（m） | | |
| 验证集 Axis-XY P90 误差（m） | | |
| Confidence 与误差相关系数 | | |
| 冻结参数检查是否 PASS | | |
| 峰值 GPU 显存 | | |

实例分割结果记录：

| 数据集/配置 | Completeness | Commission | F1 | Precision | Recall | Coverage |
|---|---:|---:|---:|---:|---:|---:|
| L1W weight 0 | | | | | | |
| L1W weight 0.1 | | | | | | |
| L1W weight 0.25 | | | | | | |
| L1W weight 0.5 | | | | | | |
| Wytham 官方 A0 | 64.8 | 18.9 | 72.0 | 62.5 | 80.5 | 57.7 |
| Wytham Point Transformer | | | | | | |

## 9. 常用故障排查

判断训练是否仍在运行：

```bash
pgrep -af "[t]ools/training/train.py"
```

判断 pipeline 是否仍在运行：

```bash
pgrep -af "[t]ools/pipeline/pipeline.py"
```

查看内存和显存：

```bash
free -h
nvidia-smi
```

查看最近日志：

```bash
tail -n 100 logs/train_axis_pt_frozen.log
tail -n 100 logs/pipeline_wytham_axis_pt_final.log
```

如果出现 CUDA OOM，先将训练配置中的：

```yaml
model:
  axis_query_chunk_size: 4096
```

当前 Point Transformer 配置已经使用 `batch_size: 1`。如果 4096 仍然 OOM，
再改为 2048。不要先修改邻域数量、体素尺寸或网络维度，因为 query 分块不改变
计算结果，而这些结构参数会使 MLP/PT 的对比协议发生变化。
