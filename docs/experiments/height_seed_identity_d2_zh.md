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

- checkpoint 必须先满足 validation seed semantic retention 至少 `99.5%`，再按
  validation selection loss 选择；
- L1W Detection F1 不低于官方 A0 的 `98.4%`；
- 原始 TreeLearn shared tensors 继续 bitwise 不变；
- D2-A2 相对 D2-A1 的 L1W F1 不得低超过 `0.3 pp`。

首轮训练结果中，D2-A1 的 eligible 最佳点为 epoch 25（retention `99.59%`）；
D2-A2 的无约束最低 loss 位于 epoch 25，但 retention 仅 `99.42%`，因此不具备
资格。D2-A2 的 eligible 最佳点为 epoch 10（retention `99.96%`、selection loss
`0.4336`）。L1W A2 配置固定读取 `epoch_10.pth`；无需重新训练，也不得用 L1W
或 Wytham 改选 epoch。

通过后才生成锁定的 Wytham D2-A1/D2-A2 配置，并只运行一次。若失败，不扫描
identity weight；记录 identity-preservation 不能同时保持分割性能，关闭本路线。
## 7. L1W 结果与锁定 Wytham

固定 checkpoint 的 L1W 结果：

| Model | Checkpoint | Completeness | Commission | F1 | Coverage |
|---|---|---:|---:|---:|---:|
| A0 official | official | 100.0% | 3.1% | 98.4% | 98.0% |
| D2-A1 Height-MLP | epoch 25 | 100.0% | 4.3% | 97.8% | 97.9% |
| D2-A2 HSCA | epoch 10 | 100.0% | 2.5% | 98.7% | 98.0% |

D2-A1 是失败的参数匹配负控制，不进入 Wytham。D2-A2 相对 A0 提升 `0.3 pp`
F1、降低 `0.6 pp` Commission，同时相对 D2-A1 提升 `0.9 pp` F1，故 proposed
model 单独满足外测资格。该决策只使用固定 validation 与 L1W regression；Wytham
checkpoint、配置及全部参数继续锁定。

一次性 Wytham 命令：

```bash
python -u tools/pipeline/pipeline.py \
  --config configs/experiments/height_context_attention/pipeline_wytham_d2_a2_seed_identity_hsca_locked.yaml \
  > logs/height_context_attention/pipeline_wytham_d2_a2_seed_identity_hsca_locked.log 2>&1

python -u tools/evaluation/evaluate.py \
  --config configs/experiments/height_context_attention/evaluate_wytham_d2_a2_seed_identity_hsca_locked.yaml \
  > logs/height_context_attention/evaluate_wytham_d2_a2_seed_identity_hsca_locked.log 2>&1
```

外测判定相对已锁定 A0（F1 `72.0%`、Completeness `64.8%`、Commission `18.9%`、
Coverage `57.7%`）：主要成功条件为 F1 至少 `72.5%`；辅助条件为 Commission 不升高、
Completeness 下降不超过 `1.0 pp`。结果无论成败均不再改 checkpoint、identity weight、
margin 或 Wytham grouping 参数。
## 8. Wytham 失败与无 GT 归因

锁定 D2-A2 epoch 10 的 Wytham 结果为：Completeness `63.5%`、Commission
`22.4%`、F1 `69.8%`、Precision `63.2%`、Recall `78.3%`、Coverage `57.1%`。
相对 A0，F1 下降 `2.2 pp`、Commission 增加 `3.5 pp`、Completeness 下降
`1.3 pp`。外测 Gate 失败；不得调整 checkpoint、identity weight、margin 或
Wytham grouping 参数。

下一步只运行不读取 GT 的失败归因：

```bash
python -u tools/diagnostics/diagnose_height_context_domain_drift.py \
  --suite d2_a2 \
  --max_scans 2 \
  --output_dir logs/height_context_attention/d3_d2_a2_drift_smoke \
  2>&1 | tee logs/height_context_attention/d3_d2_a2_drift_smoke.log

nohup python -u tools/diagnostics/diagnose_height_context_domain_drift.py \
  --suite d2_a2 \
  --output_dir logs/height_context_attention/d3_d2_a2_drift \
  > logs/height_context_attention/d3_d2_a2_drift_run.log 2>&1 < /dev/null &
```

同时从既有 pipeline 日志读取 seed 数，不重新运行 pipeline：

```bash
grep -E "Clustering .*seed points|Confidence-filtered base seeds" \
  logs/height_context_attention/pipeline_wytham_d2_a2_seed_identity_hsca_locked.log
```

固定解释规则：

- non-tree→tree 或 seed-added 的 Wytham/L1W 倍率明显升高：semantic expansion
  导致伪实例，后续只研究双向 semantic identity，不改 grouping；
- offset residual 稳定但 seed-added 不升高：失败来自训练域相关的实例拓扑，停止
  adapter 部署路线；
- offset residual 或 offset-caused seed removal 明显放大：只允许 semantic adapter，
  offset 输出保持官方值，再做一次 validation/L1W 实验；
- 任一新方案都必须重新从 validation 和 L1W 过 Gate，不能直接回到 Wytham。
