# 双锚点 / 树轴形态版本在 AutoDL 上运行

本实现直接基于原版 TreeLearn，未加入注意力模块。新增内容包括：

- 鲁棒基部锚点 `B`；
- 相对树高 0.55--0.75 处的上部形态锚点 `U`；
- 独立的 base-offset 和 upper-offset 预测头；
- `B -> U` 树轴余弦损失；
- 基部 XY、上部 XY 和轴高构成的联合聚类空间。

锚点标签由现有实例标签在线计算，因此已有的训练 crop 可以直接使用，不需要
重新生成。

## 环境

AutoDL 建议使用 Python 3.10、CUDA 11.8 和至少 24 GB 显存：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
cd /root/autodl-tmp/TreeLearn
bash setup/setup.sh
conda activate TreeLearn
```

确保 PyTorch CUDA 版本为 11.8，并使用 `spconv-cu118`。

## 数据路径

将数据放置为：

```text
/root/autodl-fs/data/
├── train/random_crops/npz/
├── val/tiles/npz/
├── val/forest/L1W.laz
└── model_weights/model_weights_with_small_20241213.pth
```

然后把以下配置中的相对路径改成上面的绝对路径：

- `configs/_modular/dataset_train.yaml`
- `configs/_modular/dataset_test.yaml`
- `configs/training/train.yaml`

## 第一次检查

在 AutoDL 上先把 `configs/training/train.yaml` 临时改为：

```yaml
epochs: 2
examples_per_epoch: 20
validation_frequency: 1

dataloader:
  train:
    batch_size: 1
```

运行：

```bash
python tools/training/train.py \
  --config configs/training/train.yaml \
  --work_dir dual_anchor_v1
```

加载原版 checkpoint 时出现 `upper_offset_linear.*` missing keys 是正常的，
因为上部锚点头是新增层。旧权重只能用于初始化，不能直接用于双锚点推理。

训练日志应包含：

```text
offset_loss
upper_offset_loss
axis_loss
val/Axis_Cosine_Error
```

确认两轮可以正常结束后，恢复正式的 epochs、examples_per_epoch 和
validation_frequency。

## 正式训练

```bash
python tools/training/train.py \
  --config configs/training/train.yaml \
  --work_dir dual_anchor_v1
```

checkpoint 默认位于：

```text
/root/autodl-tmp/TreeLearn/work_dirs/dual_anchor_v1/
```

重要权重应复制到持久化数据盘：

```bash
mkdir -p /root/autodl-fs/checkpoints/dual_anchor_v1
cp work_dirs/dual_anchor_v1/epoch_*.pth \
  /root/autodl-fs/checkpoints/dual_anchor_v1/
```

## 推理

修改 `configs/pipeline/pipeline.yaml`：

```yaml
forest_path: '/root/autodl-fs/data/val/forest/L1W.laz'
pretrain: '/root/autodl-fs/checkpoints/dual_anchor_v1/epoch_XXX.pth'
```

运行：

```bash
python tools/pipeline/pipeline.py \
  --config configs/pipeline/pipeline.yaml
```

点级结果中新增：

```text
upper_offset_predictions
upper_offset_labels
cluster_coords_upper.laz
```

## 消融设置

原版单锚点：

```yaml
model:
  use_upper_anchor: False
  offset_loss_type: 'l2'
```

双锚点但不使用树轴损失：

```yaml
model:
  use_upper_anchor: True
  axis_loss_weight: 0.0

grouping:
  axis_height_weight: 0.0
```

完整版本：

```yaml
model:
  use_upper_anchor: True
  upper_offset_loss_weight: 1.0
  axis_loss_weight: 0.1

grouping:
  upper_anchor_weight: 1.0
  axis_height_weight: 0.1
```

如果联合聚类产生较多碎片，先把 `axis_height_weight` 改为 0；仍有碎片时，
再把 `upper_anchor_weight` 从 1.0 降到 0.5。每次只修改一个参数。

