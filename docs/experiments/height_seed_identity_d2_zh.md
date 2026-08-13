# D2 Seed-Preserving Identity Height Adapter

## 1. D1 结论

全量 D1 使用 L1W 6,105,000 个采样点和 Wytham 10,045,000 个采样点。Wytham
相对 L1W 的 base-seed removal 放大为：

- Height-MLP：`1.928x`；
- HSCA：`1.711x`；
- 其中 semantic removal 分别放大 `2.916x` 和 `2.831x`；
- XY/Z residual 与 gate 仅约 `0.98-1.04x`，不存在幅度爆炸。

因此 D2 只处理 base-seed semantic identity，不增加 offset 或 gate 正则。

## 2. 方法与固定参数

冻结 TreeLearn 产生 teacher base seeds：

```text
p_base(tree) >= 0.5
verticality > 0.6
abs(base_offset_z) < 4.0 m
```

对这些点规定最低适配概率：

```text
p_min = max(p_base(tree) - 0.02, 0.501)
L_identity = mean(relu(p_min - p_adapted(tree))^2)
L_total = L_original + 50 * L_identity
```

这是单向约束：不会阻止适配器提高 tree probability，也不直接约束 non-tree 点。
`50` 与 TreeLearn 原 semantic loss multiplier 相同；本轮只运行这一组固定值。

D2-A1 和 D2-A2 使用完全相同的数据、seed、epoch、optimizer、identity loss 和
checkpoint 选择规则，唯一差异仍为 MLP mixer 与 Q/K/V attention mixer。

## 3. 服务器预检

```bash
cd ~/projects/zrx/code/TreeLearn
git pull --ff-only origin vertical-instance-quality
conda activate TreeLearn
mkdir -p logs/height_context_attention

python -m unittest tests.test_height_seed_identity -v

python tools/diagnostics/verify_height_seed_identity_setup.py \
  --config configs/experiments/height_context_attention/train_d2_a1_seed_identity_mlp.yaml \
  --reference data/model_weights/model_weights_with_small_20241213.pth

python tools/diagnostics/verify_height_seed_identity_setup.py \
  --config configs/experiments/height_context_attention/train_d2_a2_seed_identity_hsca.yaml \
  --reference data/model_weights/model_weights_with_small_20241213.pth

python tools/diagnostics/verify_height_context_setup.py \
  --config configs/experiments/height_context_attention/train_d2_a1_seed_identity_mlp.yaml \
  --reference data/model_weights/model_weights_with_small_20241213.pth \
  2>&1 | tee logs/height_context_attention/verify_d2_a1_setup.log

python tools/diagnostics/verify_height_context_setup.py \
  --config configs/experiments/height_context_attention/train_d2_a2_seed_identity_hsca.yaml \
  --reference data/model_weights/model_weights_with_small_20241213.pth \
  2>&1 | tee logs/height_context_attention/verify_d2_a2_setup.log
```

两个 setup 都必须 PASS，且原始 TreeLearn tensors 的变化数为 0。

## 4. 训练

```bash
nohup bash -c '
set -e

for experiment in d2_a1_seed_identity_mlp d2_a2_seed_identity_hsca; do
  config="configs/experiments/height_context_attention/train_${experiment}.yaml"
  echo "===== START ${experiment} $(date) ====="
  python -u tools/training/train.py \
    --config "$config" \
    > "logs/height_context_attention/train_${experiment}.log" 2>&1
  echo "===== FINISHED ${experiment} $(date) ====="
done
' > logs/height_context_attention/d2_a1_a2_runner.log 2>&1 < /dev/null &

echo $! | tee logs/height_context_attention/d2_a1_a2_runner.pid
tail -f logs/height_context_attention/d2_a1_a2_runner.log
```

训练结束后检查：

```bash
grep -E \
  "height_selection_loss|height_seed_identity_loss|height_seed_semantic_retention|Saved best" \
  logs/height_context_attention/train_d2_*.log

python tools/diagnostics/verify_height_context_checkpoint.py \
  --reference data/model_weights/model_weights_with_small_20241213.pth \
  --candidate work_dirs/train_d2_a1_seed_identity_mlp/best_height_mlp.pth

python tools/diagnostics/verify_height_context_checkpoint.py \
  --reference data/model_weights/model_weights_with_small_20241213.pth \
  --candidate work_dirs/train_d2_a2_seed_identity_hsca/best_hsca.pth
```

## 5. L1W 回归

```bash
nohup bash -c '
set -e

for experiment in d2_a1_seed_identity_mlp d2_a2_seed_identity_hsca; do
  python -u tools/pipeline/pipeline.py \
    --config "configs/experiments/height_context_attention/pipeline_l1w_${experiment}.yaml" \
    > "logs/height_context_attention/pipeline_l1w_${experiment}.log" 2>&1

  python -u tools/evaluation/evaluate.py \
    --config "configs/experiments/height_context_attention/evaluate_l1w_${experiment}.yaml" \
    > "logs/height_context_attention/evaluate_l1w_${experiment}.log" 2>&1
done
' > logs/height_context_attention/d2_l1w_runner.log 2>&1 < /dev/null &

echo $! | tee logs/height_context_attention/d2_l1w_runner.pid
tail -f logs/height_context_attention/d2_l1w_runner.log
```

汇总：

```bash
for experiment in d2_a1_seed_identity_mlp d2_a2_seed_identity_hsca; do
  echo "========== ${experiment} =========="
  grep -E \
    "Completeness:|Commission Error Rate:|F1 Score:|Precision:|Recall:|Coverage:" \
    "logs/height_context_attention/evaluate_l1w_${experiment}.log"
done
```

## 6. Gate 与下一步

进入锁定 Wytham D2 对照的必要条件：

- 最佳 checkpoint 的 validation seed semantic retention 至少 `99.5%`；
- L1W Detection F1 不低于官方 A0 的 `98.4%`；
- 原始 TreeLearn shared tensors 继续 bitwise 不变；
- D2-A2 相对 D2-A1 的 L1W F1 不得低超过 `0.3 pp`。

通过后才生成锁定的 Wytham D2-A1/D2-A2 配置，并只运行一次。若失败，不扫描
identity weight；记录 identity-preservation 不能同时保持分割性能，关闭本路线。
