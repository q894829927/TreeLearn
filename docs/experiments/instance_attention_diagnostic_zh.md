# 实例级 Attention 前置诊断

## 1. 目的

本阶段不训练 Attention。先验证 TreeLearn 预测出的真实树实例（TP）和计入 Commission 的假阳性实例（FP）是否能被已有置信度与形态特征区分。

只有 Wytham 同时满足以下条件才进入实例级 Attention：

1. 计入 Commission 的 FP 不少于 20 个；
2. 最佳 confidence 单特征的 TP/FP separability ROC-AUC 不低于 0.75。

如果失败，停止实例 Attention，不继续堆模块。

## 2. 诊断特征

pipeline 不保存大体积逐点 NPZ，只为每个预测实例保存一行 CSV，包括：

- 点数、聚类 seed 数和 seed 比例；
- confidence 与 seed confidence 的 mean/std/P10/P50/P90；
- semantic tree probability 的 mean/std/P10/P50/P90；
- verticality 的 mean/std/P10/P50/P90；
- 高度、XY 尺寸、包围盒面积、点密度；
- XY 半径、base-vote 紧凑度和 seed base-vote 紧凑度；
- XY offset 幅度与 Z offset 绝对值统计。

诊断聚类使用 r100，即完全不筛 seed。MLP residual 不参与坐标融合，confidence 只被记录，不影响聚类。

## 3. 运行诊断 pipeline

先确认正式 checkpoint 和既有 baseline 评估结果存在：

```bash
test -f work_dirs/base_residual_mlp_frozen_s43/best_base_residual_xy.pth
test -f data/pipeline/L1W/results_seed_ratio_r100/full_forest/evaluation/evaluation_results.pt
test -f data/pipeline/wytham/results_seed_conf_t000_control/full_forest/evaluation/evaluation_results.pt
```

顺序运行 L1W 和 Wytham：

```bash
mkdir -p logs

nohup bash -c '
set -e

echo ===== START L1W instance diagnostic =====
python -u tools/pipeline/pipeline.py \
  --config configs/experiments/point_transformer/pipeline_l1w_instance_diagnostic_mlp_s43_r100.yaml \
  > logs/pipeline_l1w_instance_diagnostic.log 2>&1

echo ===== START Wytham instance diagnostic =====
python -u tools/pipeline/pipeline.py \
  --config configs/experiments/point_transformer/pipeline_wytham_instance_diagnostic_mlp_s43_r100.yaml \
  > logs/pipeline_wytham_instance_diagnostic.log 2>&1

echo ===== FINISHED instance diagnostic pipelines =====
' > logs/instance_diagnostic_runner.log 2>&1 < /dev/null &

echo $! | tee logs/instance_diagnostic_runner.pid
tail -f logs/instance_diagnostic_runner.log
```

输出应位于：

```text
data/pipeline/L1W/results_instance_diagnostic_mlp_s43_r100/instance_diagnostics/instance_features.csv
data/pipeline/wytham/results_instance_diagnostic_mlp_s43_r100/instance_diagnostics/instance_features.csv
```

## 4. 验证诊断聚类与 baseline 分区完全相同

HDBSCAN 可能为完全相同的聚类分区分配不同的数字 ID，因此不能直接使用
`np.array_equal` 比较原始标签。第 5 节的 AUC 工具会自动完成更严格的验证：

1. 检查点数和逐点坐标一致；
2. 建立 diagnostic ID 到 baseline ID 的一一映射；
3. 映射后要求每个点的实例标签完全一致；
4. 验证成功后，把 CSV 中的实例 ID 映射为 baseline ID，再复用既有 evaluation。

输出中必须出现：

```text
PASS: predicted-instance partitions are identical after a one-to-one label-ID remapping.
```

如果点集分区真的变化，工具会报出 differing points 并停止。此时不能复用旧
evaluation，必须重新评估诊断预测。

为了避免 HDBSCAN 重复运行时的少量非确定性，推荐直接为 diagnostic 输出运行
一次匹配评估：

```bash
python -u tools/evaluation/evaluate.py \
  --config configs/experiments/point_transformer/evaluate_l1w_instance_diagnostic_mlp_s43_r100.yaml \
  2>&1 | tee logs/evaluate_l1w_instance_diagnostic_mlp_s43_r100.log

python -u tools/evaluation/evaluate.py \
  --config configs/experiments/point_transformer/evaluate_wytham_instance_diagnostic_mlp_s43_r100.yaml \
  2>&1 | tee logs/evaluate_wytham_instance_diagnostic_mlp_s43_r100.log
```

## 5. 计算 TP/FP 特征 AUC

```bash
python tools/diagnostics/diagnose_instance_separability.py \
  --features data/pipeline/L1W/results_instance_diagnostic_mlp_s43_r100/instance_diagnostics/instance_features.csv \
  --evaluation data/pipeline/L1W/results_instance_diagnostic_mlp_s43_r100/full_forest/evaluation/evaluation_results.pt \
  --output_dir logs/instance_separability_l1w \
  2>&1 | tee logs/instance_separability_l1w.log

python tools/diagnostics/diagnose_instance_separability.py \
  --features data/pipeline/wytham/results_instance_diagnostic_mlp_s43_r100/instance_diagnostics/instance_features.csv \
  --evaluation data/pipeline/wytham/results_instance_diagnostic_mlp_s43_r100/full_forest/evaluation/evaluation_results.pt \
  --output_dir logs/instance_separability_wytham \
  2>&1 | tee logs/instance_separability_wytham.log
```

重点查看：

```bash
cat logs/instance_separability_wytham/summary.md
cat logs/instance_separability_wytham/summary.json
```

L1W baseline 只有 5 个 FP，因此 L1W AUC 仅作参考；是否继续主要依据 Wytham。

## 6. 继续/停止规则

- Wytham 输出 `PASS`：实现 mean-pooling MLP 实例验证器，再实现 Attention pooling；Attention 必须超过 MLP 才能作为创新点。
- Wytham 输出 `STOP`，但最佳 overall feature AUC ≥ 0.80：先研究简单规则/MLP，不直接实现 Attention。
- Wytham confidence 和 overall AUC 都较低：停止该路线，当前论文只保留置信度 seed 筛选。

Wytham 已用于开发诊断；本阶段之后不能把它描述成完全未见测试，也不能根据其结果反复调阈值。
