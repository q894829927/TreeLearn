# 冻结 TreeLearn 主干的 Point Transformer 树轴实验

## 1. 当前结论

旧冠层锚点实验已经完成：

| 实验 | 最佳 epoch | Mean（m） | Median（m） | P90（m） |
|---|---:|---:|---:|---:|
| MLP + 异方差损失 | 5 | 1.492 | 1.334 | 2.691 |
| Point Transformer + 异方差损失 | 25 | 1.471 | 1.311 | 2.674 |
| Point Transformer + 解耦损失 | 25 | 1.480 | 1.323 | 2.684 |

Point Transformer 相对 MLP 的 Mean 只降低约 1.4%，没有达到预定的 5% 门槛；换成解耦损失也没有改善。因此暂停 L1W 和 Wytham 测试，先检查并修改 Axis-XY 标签，不再增加网络层数。

新标签使用树干下部拟合轴线：

```text
树高 10%～50% 范围
        ↓
verticality >= 0.60 的树干候选点
        ↓
沿高度划分 4 个区间，每区间取 XY 中位数
        ↓
至少 3 个有效区间
        ↓
Theil–Sen 稳健直线拟合
        ↓
外推到 65% 树高，得到 upper anchor
        ↓
Axis-XY = upper anchor XY - base anchor XY
```

旧配置默认仍使用 `upper_anchor_mode: crown_median`，不会改变已经完成的实验。新实验显式使用 `upper_anchor_mode: stem_axis`。

## 2. 固定实验协议

所有实验从以下官方 checkpoint 初始化：

```text
data/model_weights/model_weights_with_small_20241213.pth
```

始终冻结：

```text
input_conv
unet
output_layer
semantic_linear
offset_linear
```

只训练：

```text
axis_input_projection
axis_point_transformer（仅 PT 组）
axis_xy_head
axis_uncertainty_head
```

Wytham 是最终外部测试集，不允许用于选择 checkpoint、标签参数或融合权重。

## 3. 第一步：环境和代码检查

在 Linux 服务器的 TreeLearn 根目录运行：

```bash
conda activate TreeLearn
mkdir -p logs
git branch --show-current

python -m unittest discover -s tests -p 'test_stem_axis_labels.py' -v
python -m unittest discover -s tests -p 'test_axis_point_transformer.py' -v
```

分支应为：

```text
point-transformer
```

检查配置：

```bash
python - <<'PY'
from tree_learn.util import get_config

files = [
    "configs/experiments/point_transformer/train_axis_mlp_stem_axis.yaml",
    "configs/experiments/point_transformer/train_axis_pt_stem_axis.yaml",
    "configs/experiments/point_transformer/pipeline_l1w_axis_pt_stem_w000.yaml",
    "configs/experiments/point_transformer/pipeline_wytham_axis_pt_stem_final.yaml",
]

for path in files:
    cfg = get_config(path)
    print(path, "OK")
    if hasattr(cfg, "dataset_train"):
        print("  upper anchor:", cfg.dataset_train.upper_anchor_mode)
    if hasattr(cfg, "model"):
        print("  branch:", cfg.model.axis_branch_type)
PY

test -f data/model_weights/model_weights_with_small_20241213.pth
```

## 4. 第二步：先诊断标签，不启动训练

运行全部 323 个验证 tile 的标签诊断：

```bash
python tools/diagnostics/diagnose_stem_axis_labels.py \
  --config configs/experiments/point_transformer/train_axis_mlp_stem_axis.yaml \
  --max_scans 323 \
  --output_dir logs/stem_axis_diagnostic \
  2>&1 | tee logs/diagnose_stem_axis_labels.log
```

输出：

```text
logs/stem_axis_diagnostic/summary.json
logs/stem_axis_diagnostic/label_examples.png
```

同时满足以下条件才继续：

1. `stem_to_crown_coverage_ratio >= 0.70`；
2. `stem_axis_xy_length.mean >= 0.30 m`；
3. 查看 `label_examples.png`，蓝色树干轴锚点应沿树干倾斜方向，不能明显落到相邻树冠。

若日志输出 `STOP`，不要训练 MLP/PT。此时优先降低 `stem_axis_verticality_threshold` 到 `0.50`，重新诊断一次；仍失败则说明现有 verticality 或点密度不足，需要重新设计标签。

## 5. 第三步：只跑 stem-axis MLP 控制组

标签诊断通过后运行：

```bash
nohup python -u tools/training/train.py \
  --config configs/experiments/point_transformer/train_axis_mlp_stem_axis.yaml \
  --work_dir axis_mlp_stem_axis \
  > logs/train_axis_mlp_stem_axis.log 2>&1 < /dev/null &

echo $! | tee logs/train_axis_mlp_stem_axis.pid
tail -f logs/train_axis_mlp_stem_axis.log
```

查看结果：

```bash
grep "axis_xy_mean_error" logs/train_axis_mlp_stem_axis.log
grep "Saved best_axis_xy" logs/train_axis_mlp_stem_axis.log

python tools/diagnostics/verify_axis_checkpoint.py \
  --reference data/model_weights/model_weights_with_small_20241213.pth \
  --candidate work_dirs/axis_mlp_stem_axis/best_axis_xy.pth
```

验证日志新增两个与标签尺度有关的指标：

- `axis_target_xy_mean_length`：目标 Axis-XY 的平均长度；
- `axis_normalized_mean_error`：Mean Error / Mean Target Length。

只有同时满足以下条件才运行 PT：

1. 冻结参数检查输出 `PASS`；
2. 最佳 `axis_normalized_mean_error <= 0.80`。

归一化误差为 1 表示效果接近始终预测零向量。MLP 若不能低于 0.80，说明新标签仍缺乏可学习信号，此时不应运行更昂贵的 PT。

## 6. 第四步：运行 stem-axis Point Transformer

MLP 通过门槛后运行：

```bash
nohup env PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
  python -u tools/training/train.py \
  --config configs/experiments/point_transformer/train_axis_pt_stem_axis.yaml \
  --work_dir axis_pt_stem_axis \
  > logs/train_axis_pt_stem_axis.log 2>&1 < /dev/null &

echo $! | tee logs/train_axis_pt_stem_axis.pid
tail -f logs/train_axis_pt_stem_axis.log
```

监控：

```bash
pgrep -af "[t]ools/training/train.py"
nvidia-smi
tail -n 50 logs/train_axis_pt_stem_axis.log
```

训练完成后：

```bash
grep "axis_xy_mean_error" logs/train_axis_pt_stem_axis.log
grep "Saved best_axis_xy" logs/train_axis_pt_stem_axis.log

python tools/diagnostics/verify_axis_checkpoint.py \
  --reference data/model_weights/model_weights_with_small_20241213.pth \
  --candidate work_dirs/axis_pt_stem_axis/best_axis_xy.pth
```

PT 必须同时满足：

1. PT 最佳 Mean ≤ MLP 最佳 Mean × 0.95；
2. PT 最佳 P90 ≤ MLP 最佳 P90；
3. PT 最佳归一化 Mean ≤ 0.80；
4. 冻结参数检查输出 `PASS`。

任一条件失败，停止 L1W/Wytham，不继续增加注意力层。

## 7. 第五步：L1W 选择融合权重

PT 达标后测试固定权重 `0、0.1、0.25、0.5`：

```bash
for weight in 000 010 025 050; do
  nohup python -u tools/pipeline/pipeline.py \
    --config configs/experiments/point_transformer/pipeline_l1w_axis_pt_stem_w${weight}.yaml \
    > logs/pipeline_l1w_axis_pt_stem_w${weight}.log 2>&1 < /dev/null &

  pid=$!
  echo "$pid" | tee logs/pipeline_l1w_axis_pt_stem_w${weight}.pid
  wait "$pid" || exit 1

  python -u tools/evaluation/evaluate.py \
    --config configs/experiments/point_transformer/evaluate_l1w_axis_pt_stem_w${weight}.yaml \
    2>&1 | tee logs/evaluate_l1w_axis_pt_stem_w${weight}.log
done
```

选择规则：

1. Detection F1 最高；
2. F1 相同时选择 Coverage 更高者；
3. 两者仍相同时选择更小权重。

`w000` 必须与 base-only 2D 聚类一致。四组配置默认 `tile_generation: False`，复用完全相同的 L1W tiles。

## 8. 第六步：Wytham 只运行一次

将 L1W 选定的唯一权重写入：

```text
configs/experiments/point_transformer/pipeline_wytham_axis_pt_stem_final.yaml
```

只修改：

```yaml
grouping:
  axis_fusion_weight: 选定权重
```

然后运行：

```bash
nohup python -u tools/pipeline/pipeline.py \
  --config configs/experiments/point_transformer/pipeline_wytham_axis_pt_stem_final.yaml \
  > logs/pipeline_wytham_axis_pt_stem_final.log 2>&1 < /dev/null &

echo $! | tee logs/pipeline_wytham_axis_pt_stem_final.pid
tail -f logs/pipeline_wytham_axis_pt_stem_final.log
```

完成后评估：

```bash
python -u tools/evaluation/evaluate.py \
  --config configs/experiments/point_transformer/evaluate_wytham_axis_pt_stem_final.yaml \
  2>&1 | tee logs/evaluate_wytham_axis_pt_stem_final.log
```

官方 A0 参考值：

| F1 | Precision | Recall | Coverage |
|---:|---:|---:|---:|
| 72.0% | 62.5% | 80.5% | 57.7% |

进入论文消融的门槛：

- F1 至少提升 0.5 个百分点；或
- Coverage 至少提升 1.0 个百分点，且 Precision 下降不超过 1.0 个百分点。

## 9. 结果记录表

| 项目 | Stem MLP | Stem PT |
|---|---:|---:|
| Git commit | | |
| 最佳 checkpoint | | |
| 最佳 epoch | | |
| Mean error（m） | | |
| Median error（m） | | |
| P90 error（m） | | |
| Target mean length（m） | | |
| Normalized mean error | | |
| Confidence/error correlation | | |
| 冻结检查 | | |
| GPU 与峰值显存 | | |

| L1W 权重 | Completeness | Commission | F1 | Precision | Recall | Coverage |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | | | | | | |
| 0.1 | | | | | | |
| 0.25 | | | | | | |
| 0.5 | | | | | | |

## 10. 常用排查

```bash
pgrep -af "[t]ools/training/train.py"
pgrep -af "[t]ools/pipeline/pipeline.py"
free -h
nvidia-smi
tail -n 100 logs/train_axis_pt_stem_axis.log
```

如果 PT 出现 CUDA OOM，只把 `axis_query_chunk_size` 从 `8192` 改为 `4096`，仍不足再改为 `2048`。不要先改邻居数、体素大小或隐藏维度，否则 MLP/PT 的对比协议会改变。
