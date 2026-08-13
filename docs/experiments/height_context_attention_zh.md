# HSCA 高度分层上下文注意力实验

> 状态：第一阶段代码与 A0/A1/A2 配置已建立；尚未使用 Wytham。
> 分支：`vertical-instance-quality`
> 前置结论：Q6a 恢复数量 Gate 失败，fragmentation merge 已关闭。

## 1. 研究假设

复杂森林中的树干遮挡、冠层重叠和复层结构会让逐点 sparse U-Net 特征缺少明确的
垂直上下文。HSCA 不替换 TreeLearn 主干，而是在冻结的逐点 `backbone_feats` 与
原始 semantic/base-offset heads 之后，学习一个轻量残差：

```text
冻结 TreeLearn 主干与原始 heads
        ↓
逐点 32D backbone feature + verticality + 归一化预测高度
        ↓
8 个高度层；每层 mean/max pooling
        ↓
A1：逐 token 参数匹配 MLP
A2：单层 4-head Q/K/V self-attention
        ↓
高度上下文回映射到逐点特征 + sigmoid gate
        ↓
semantic residual + base-offset residual
```

A1 与 A2 除 token mixer 外完全相同。默认配置下：

- A1 Height-MLP：9,831 个可训练参数；
- A2 HSCA：9,863 个可训练参数；
- 差异：32 个参数，即约 0.33%。

因此 A2 若稳定优于 A1，才能把收益归因于跨高度层注意力，而不是简单增加参数。

## 2. 固定设计

- `height_num_bins=8`；
- `height_hidden_dim=32`；
- `height_num_heads=4`；
- `height_dropout=0.1`；
- 高度代理：`relu(-frozen_offset_z)`；
- 每个 tile 独立按 P95 归一化，并限制在 `[0,1]`；
- 空高度层使用 `key_padding_mask`；
- 原始 `input_conv/unet/output_layer/semantic_linear/offset_linear` 永久冻结；
- 不启用 upper anchor、axis branch、seed confidence filter 或 Wytham 参数扫描；
- residual heads 零初始化，residual scale 初始化为 1。

最后一项是有意修正：如果 residual head 和 scale 同时为 0，两者第一步梯度都为 0，
分支无法启动。当前初始化仍保证初始 semantic/offset 输出与官方 TreeLearn bitwise
一致，同时 residual head 在第一步能得到梯度。

## 3. 新增文件

```text
tree_learn/model/height_context_attention.py
tests/test_height_context_attention.py
tools/diagnostics/verify_height_context_setup.py
tools/diagnostics/verify_height_context_checkpoint.py

configs/experiments/height_context_attention/
  _train_height_adapter_common.yaml
  train_a1_height_mlp_frozen.yaml
  train_a2_hsca_frozen.yaml
  _pipeline_l1w_common.yaml
  pipeline_l1w_a0_official.yaml
  pipeline_l1w_a1_height_mlp.yaml
  pipeline_l1w_a2_hsca.yaml
  evaluate_l1w_a0_official.yaml
  evaluate_l1w_a1_height_mlp.yaml
  evaluate_l1w_a2_hsca.yaml
```

训练器使用 323 个固定 validation tiles 的
`50 × semantic_loss + offset_loss` 选择 checkpoint：

- A1：`best_height_mlp.pth`；
- A2：`best_hsca.pth`。

这个选择过程不读取 Wytham，也不使用 L1W 最终检测指标挑 epoch。

## 4. 服务器代码检查

拉取代码后先执行：

```bash
cd ~/projects/zrx/code/TreeLearn
conda activate TreeLearn
mkdir -p logs/height_context_attention

python -m unittest tests.test_height_context_attention -v

python tools/diagnostics/verify_height_context_setup.py \
  --config configs/experiments/height_context_attention/train_a1_height_mlp_frozen.yaml \
  --reference data/model_weights/model_weights_with_small_20241213.pth \
  2>&1 | tee logs/height_context_attention/verify_a1_setup.log

python tools/diagnostics/verify_height_context_setup.py \
  --config configs/experiments/height_context_attention/train_a2_hsca_frozen.yaml \
  --reference data/model_weights/model_weights_with_small_20241213.pth \
  2>&1 | tee logs/height_context_attention/verify_a2_setup.log
```

两个诊断都必须输出：

```text
changed shared tensors during load: 0
changed original tensors after one step: 0
PASS: exact initialization and frozen optimizer scope verified.
```

若任一检查失败，停止训练并先修复代码。

## 5. 第一轮训练：只跑 A1 与 A2 的 seed 42

```bash
nohup bash -c '
set -e

for experiment in a1_height_mlp a2_hsca; do
  if [ "$experiment" = "a1_height_mlp" ]; then
    config="configs/experiments/height_context_attention/train_a1_height_mlp_frozen.yaml"
  else
    config="configs/experiments/height_context_attention/train_a2_hsca_frozen.yaml"
  fi

  echo "===== START ${experiment} $(date) ====="
  python -u tools/training/train.py \
    --config "$config" \
    > "logs/height_context_attention/train_${experiment}.log" 2>&1
  echo "===== FINISHED ${experiment} $(date) ====="
done
' > logs/height_context_attention/a1_a2_runner.log 2>&1 < /dev/null &

echo $! | tee logs/height_context_attention/a1_a2_runner.pid
tail -f logs/height_context_attention/a1_a2_runner.log
```

训练参数固定为 batch size 2、AdamW、lr `1e-3`、weight decay `1e-3`、warmup 5
个 epoch、总计 50 epochs、每 5 epochs 验证。A1/A2 的数据、随机种子、训练步数和
checkpoint 选择标准完全相同。

监控：

```bash
pgrep -af "[t]ools/training/train.py"
watch -n 2 nvidia-smi

tail -f logs/height_context_attention/train_a1_height_mlp.log
tail -f logs/height_context_attention/train_a2_hsca.log
```

训练完成后：

```bash
grep -E \
  "height_selection_loss|Saved best_height_mlp|Saved best_hsca" \
  logs/height_context_attention/train_*.log

python tools/diagnostics/verify_height_context_checkpoint.py \
  --reference data/model_weights/model_weights_with_small_20241213.pth \
  --candidate work_dirs/train_a1_height_mlp_frozen/best_height_mlp.pth

python tools/diagnostics/verify_height_context_checkpoint.py \
  --reference data/model_weights/model_weights_with_small_20241213.pth \
  --candidate work_dirs/train_a2_hsca_frozen/best_hsca.pth
```

## 6. L1W 回归检查

L1W 只用于确认 pipeline 集成与源域性能没有明显倒退，不用于从多个 HSCA 超参数中
挑最好结果。

```bash
nohup bash -c '
set -e

for experiment in a0_official a1_height_mlp a2_hsca; do
  echo "===== START pipeline ${experiment} $(date) ====="
  python -u tools/pipeline/pipeline.py \
    --config "configs/experiments/height_context_attention/pipeline_l1w_${experiment}.yaml" \
    > "logs/height_context_attention/pipeline_l1w_${experiment}.log" 2>&1

  echo "===== START evaluation ${experiment} $(date) ====="
  python -u tools/evaluation/evaluate.py \
    --config "configs/experiments/height_context_attention/evaluate_l1w_${experiment}.yaml" \
    > "logs/height_context_attention/evaluate_l1w_${experiment}.log" 2>&1
done
' > logs/height_context_attention/l1w_a0_a1_a2_runner.log 2>&1 < /dev/null &

echo $! | tee logs/height_context_attention/l1w_a0_a1_a2_runner.pid
tail -f logs/height_context_attention/l1w_a0_a1_a2_runner.log
```

汇总：

```bash
for experiment in a0_official a1_height_mlp a2_hsca; do
  echo "========== ${experiment} =========="
  grep -E \
    "Completeness:|Commission Error Rate:|F1 Score:|Precision:|Recall:|Coverage:" \
    "logs/height_context_attention/evaluate_l1w_${experiment}.log"
done
```

A2 的 L1W Detection F1 相对 A0 下降超过 `0.5 pp` 时，不进入复杂林完整验证。

## 7. 下一阶段 Gate

完成 A1/A2 seed 42 和 L1W 回归后，再生成固定五个完整 validation forests 的
A0/A1/A2 pipeline。不得提前运行 Wytham。

继续条件：

- A2 相对 A1 的五森林 macro Detection F1 至少 `+0.3 pp`，或 Coverage 至少
  `+0.5 pp`；
- 至少 3/5 个森林同方向提升；
- L1W 相对 A0 的 Detection F1 下降不超过 `0.5 pp`。

若通过：运行 seed 43/44，并设计参数匹配的 SODA-only 与 HSCA+SODA。
若失败：记录真实注意力负消融，停止扩大 HSCA，不得去 Wytham 选择参数。
