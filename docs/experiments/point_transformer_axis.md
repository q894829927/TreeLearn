# 冻结 TreeLearn 主干的 Point Transformer 树轴实验

## 0. 2026-07-30 当前结论与下一阶段

下部树干 Axis-XY 在 L1W 上的融合权重实验已经完成：

| 融合权重 | Detection F1 | Precision | Recall | Coverage |
|---:|---:|---:|---:|---:|
| 0 | 98.4227% | 98.915% | 99.070% | 98.023% |
| 0.1 | 98.4227% | 98.907% | 99.064% | 98.010% |
| 0.25 | 98.4227% | 98.894% | 99.056% | 97.991% |
| 0.5 | 98.4227% | 98.867% | 99.038% | 97.949% |

权重 `0` 胜出。直接把 base vote 沿 base→upper 轴线平移没有改变检测结果，
并且随着权重增加，分割 Coverage 单调下降。因此：

- 不使用 stem-axis 结果运行 Wytham；
- 不再继续放大 axis 融合权重；
- 不删除旧实验，保留为“直接树轴平移无效”的负结果；
- 下一阶段改为预测冻结 base-offset 的 XY 残差。

新目标为：

```text
base_residual_xy_target
  = ground_truth_base_offset_xy - frozen_base_offset_prediction_xy

refined_base_vote_xy
  = point_xy
  + frozen_base_offset_prediction_xy
  + fusion_weight × confidence × predicted_base_residual_xy
```

这样 Point Transformer 学习的是“把已有 base vote 拉回真实树基位置”，而不是把
base vote 推向树冠或上部树干。旧配置默认
`axis_target_mode: upper_axis`，新配置显式使用
`axis_target_mode: base_residual`，两类实验互不影响。

### 0.1 现在只运行 MLP 控制组

先同步本次代码，然后在服务器仓库根目录运行：

```bash
conda activate TreeLearn
mkdir -p logs

nohup python -u tools/training/train.py \
  --config configs/experiments/point_transformer/train_base_residual_mlp_frozen.yaml \
  --work_dir base_residual_mlp_frozen \
  > logs/train_base_residual_mlp_frozen.log 2>&1 < /dev/null &

echo $! | tee logs/train_base_residual_mlp_frozen.pid
tail -f logs/train_base_residual_mlp_frozen.log
```

训练时检查：

```bash
grep "base_residual_xy_mean_error" \
  logs/train_base_residual_mlp_frozen.log

grep "Saved best_base_residual_xy" \
  logs/train_base_residual_mlp_frozen.log

python tools/diagnostics/verify_axis_checkpoint.py \
  --reference data/model_weights/model_weights_with_small_20241213.pth \
  --candidate work_dirs/base_residual_mlp_frozen/best_base_residual_xy.pth
```

此前同一验证集上冻结 base head 的 XY 平均误差约为 `0.333 m`。MLP 只有同时满足
以下条件才算通过：

1. 最佳 Mean `< 0.333 m`；
2. 推荐门槛为 Mean `<= 0.316 m`，即至少降低约 5%；
3. P90 不高于未校正 base head；
4. checkpoint 冻结检查输出 `PASS`。

如果 MLP 连 `0.333 m` 都无法低于，立即停止，不训练 PT，也不跑 L1W/Wytham。

本次 MLP 的最佳结果出现在 epoch 15：

| Mean | Median | P90 | Confidence/Error correlation |
|---:|---:|---:|---:|
| 0.332 m | 0.233 m | 0.545 m | -0.287 |

Mean 相对 base-only 只改善约 0.3%，没有达到 5% 门槛。不过置信度分桶显示最高
置信度组误差为 0.206 m、最低置信度组为 0.526 m，因此在最终停止前，只对现有
checkpoint 做一次门控残差诊断，不重新训练：

```bash
python tools/diagnostics/evaluate_base_residual_fusion.py \
  --config configs/experiments/point_transformer/train_base_residual_mlp_frozen.yaml \
  --checkpoint work_dirs/base_residual_mlp_frozen/best_base_residual_xy.pth \
  --weights 0 0.25 0.5 0.75 1.0 1.5 \
  --output logs/base_residual_mlp_fusion.json \
  2>&1 | tee logs/base_residual_mlp_fusion.log
```

该工具直接计算：

```text
error(weight)
  = || target_base_residual
       - weight × confidence × predicted_base_residual ||
```

判断规则：

- 最佳门控 Mean 相对 `weight=0` 改善不足 1%：停止残差路线；
- 改善 1%～5%：暂不跑 PT，先检查误差按树高、树大小的分层分布；
- 改善至少 5% 且 P90 不恶化：才运行 PT；
- 这一步仍然只使用 323 个验证 tiles，不运行 L1W 或 Wytham。

实际门控结果为：

| Base Mean | 最佳门控 Mean | 最佳权重 | 相对改善 |
|---:|---:|---:|---:|
| 0.333029 m | 0.330855 m | 1.0 | 0.653% |

该结果低于 1% 门槛，因此停止 residual vote 修正，不运行 residual PT。

### 0.2 下一阶段：置信度引导的 base-seed 筛选

残差方向本身不足以修改 vote，但 confidence 与误差具有稳定负相关。新阶段保持
base vote、HDBSCAN 和剩余点分配不变，只筛选初始 base seeds：

```text
semantic tree
  ∩ verticality > 0.6
  ∩ |base offset-z| < 4
        ↓
confidence >= threshold
        ↓
HDBSCAN
        ↓
所有未分配树点照常最近邻分配
```

首先重新运行验证诊断，新增参数只统计实际 base seeds：

```bash
python tools/diagnostics/evaluate_base_residual_fusion.py \
  --config configs/experiments/point_transformer/train_base_residual_mlp_frozen.yaml \
  --checkpoint work_dirs/base_residual_mlp_frozen/best_base_residual_xy.pth \
  --weights 0 1 \
  --seed_confidence_thresholds 0 0.5 0.65 0.75 \
  --tree_conf_thresh 0.5 \
  --tau_vert 0.6 \
  --tau_off 4 \
  --output logs/base_seed_confidence_diagnostic.json \
  2>&1 | tee logs/base_seed_confidence_diagnostic.log
```

查看 seed 结果：

```bash
python - <<'PY'
import json

with open(
    "logs/base_seed_confidence_diagnostic.json",
    encoding="utf-8",
) as file:
    result = json.load(file)["actual_base_seeds"]

print("seed count:", result["count"])
print("confidence/error corr:", result["confidence_error_corr"])
for threshold, metrics in result["thresholds"].items():
    print(
        threshold,
        "retained:", f'{100 * metrics["retained_rate"]:.2f}%',
        "mean:", metrics["mean_base_xy_error"],
        "p90:", metrics["p90_base_xy_error"],
    )
PY
```

只有同时满足以下条件才运行 L1W：

1. seed confidence/error correlation `<= -0.15`；
2. 至少一个非零阈值保留 `>= 25%` 的 seeds；
3. 该阈值下 seed Mean error 相对阈值 0 降低 `>= 15%`。

高阈值诊断结果：

| 阈值 | Seed 保留率 | Mean（m） | Mean 改善 | P90（m） |
|---:|---:|---:|---:|---:|
| 0 | 100.00% | 0.239927 | 0% | 0.391775 |
| 0.75 | 95.10% | 0.222086 | 7.44% | 0.373260 |
| 0.80 | 81.24% | 0.208510 | 13.09% | 0.356280 |
| 0.825 | 59.47% | 0.198505 | 17.26% | 0.342612 |
| 0.85 | 21.44% | 0.186489 | 22.27% | 0.328610 |

`0.825` 同时满足保留率和误差门槛，作为主候选；`0.80` 是较保守的邻近组；
`0.85` 是低于 25% 保留率的边界压力测试。L1W 依次运行
`0、0.80、0.825、0.85`：

```bash
nohup bash -c '
set -e

for threshold in 000 080 0825 085; do
  echo "===== START pipeline t${threshold} $(date) ====="

  python -u tools/pipeline/pipeline.py \
    --config configs/experiments/point_transformer/pipeline_l1w_seed_conf_t${threshold}.yaml \
    > logs/pipeline_l1w_seed_conf_t${threshold}.log 2>&1

  echo "===== START evaluation t${threshold} $(date) ====="

  python -u tools/evaluation/evaluate.py \
    --config configs/experiments/point_transformer/evaluate_l1w_seed_conf_t${threshold}.yaml \
    > logs/evaluate_l1w_seed_conf_t${threshold}.log 2>&1

  echo "===== FINISHED t${threshold} $(date) ====="
done
' > logs/l1w_seed_conf_sweep.log 2>&1 < /dev/null &

echo $! | tee logs/l1w_seed_conf_sweep.pid
tail -f logs/l1w_seed_conf_sweep.log
```

`t000` 必须逐项复现 base-only 结果。候选阈值按 Detection F1、Coverage、较小
阈值的顺序选择。`t085` 若降低 Completeness，应直接淘汰。只有 L1W 的 F1 至少
提升 0.1 个百分点，或者 Coverage 至少提升 0.2 个百分点且 F1 不下降，才准备
一次 Wytham 最终测试。

### 0.3 历史备用方案：MLP 通过后训练 Point Transformer

```bash
nohup env PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
  python -u tools/training/train.py \
  --config configs/experiments/point_transformer/train_base_residual_pt_frozen.yaml \
  --work_dir base_residual_pt_frozen \
  > logs/train_base_residual_pt_frozen.log 2>&1 < /dev/null &

echo $! | tee logs/train_base_residual_pt_frozen.pid
tail -f logs/train_base_residual_pt_frozen.log
```

训练完成后：

```bash
grep "base_residual_xy_mean_error" \
  logs/train_base_residual_pt_frozen.log

grep "Saved best_base_residual_xy" \
  logs/train_base_residual_pt_frozen.log

python tools/diagnostics/verify_axis_checkpoint.py \
  --reference data/model_weights/model_weights_with_small_20241213.pth \
  --candidate work_dirs/base_residual_pt_frozen/best_base_residual_xy.pth
```

PT 必须满足：

1. Mean 比 MLP 最佳 Mean 至少降低 2%；理想门槛为 5%；
2. P90 不恶化；
3. 冻结检查输出 `PASS`。

只有 PT 通过后才生成 L1W 的残差融合权重配置，并测试
`{0, 0.5, 1.0, 1.5}`。权重在 L1W 锁定后，Wytham 仍只运行一次。

## 1. 历史结论：冠层与下部树干 Axis-XY

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
