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
## 9. D4 Teacher-Anchored Seed Projection

D3 无 GT 归因显示 Wytham/L1W 的 non-tree→tree rate 放大 `1.761x`、seed-added
rate 放大 `2.845x`，而 XY residual、Z residual、gate 分别仅 `0.998x`、`1.044x`、
`0.990x`。因此固定诊断指向 semantic expansion，不支持修改 offset 或 attention。

D4 不重训、不读取 Wytham GT，也不改变 HDBSCAN。对满足 frozen base 或 adapted
offset 任一几何 seed 条件的点，保留 adapted logits 的均值，但将二分类 logit margin
投影到 frozen TreeLearn 原类别一侧：

```text
m = logit_tree - logit_non_tree
base tree:     m_projected >= boundary + 0.001
base non-tree: m_projected <= boundary - 0.001
```

投影在重叠 tile 的 frozen/adapted logits 与 offset 完成 NumPy ensemble 后再次统一执行，
而不是只在单 tile 内执行；pipeline 会断言最终候选区的 teacher semantic membership
mismatch 为 0。区外点与 HSCA offset 完全不变。这从机制上阻断 semantic flip
导致的 seed 新增/删除，同时保留注意力生成的 offset 修正。先只运行 L1W：

```bash
python -m unittest tests.test_height_seed_identity -v

python -u tools/pipeline/pipeline.py \
  --config configs/experiments/height_context_attention/pipeline_l1w_d4_seed_projection_hsca.yaml \
  > logs/height_context_attention/pipeline_l1w_d4_seed_projection_hsca.log 2>&1

python -u tools/evaluation/evaluate.py \
  --config configs/experiments/height_context_attention/evaluate_l1w_d4_seed_projection_hsca.yaml \
  > logs/height_context_attention/evaluate_l1w_d4_seed_projection_hsca.log 2>&1
```

D4 L1W Gate：F1 不低于 A0 `98.4%`、Commission 不高于 `3.1%`、Completeness
保持 `100%`。失败则关闭 height-adapter 部署路线；通过后先做 L1W pointwise/seed-set
等价审计，再决定是否把 Wytham 明确降级为 development evaluation。不得直接创建
新的 Wytham 配置。
## 10. D4 结果与 D5 Full Seed Identity 收尾实验

D4 的 post-ensemble semantic membership mismatch 为 0，但 L1W seed 数由 A0 的
325,596 降为 312,584，减少 13,012（约 4.0%）。最终 F1 为 97.8%，Commission
为 4.3%，未通过 D4 Gate。由于 semantic 已严格受控，剩余 seed drift 来自
adapted offset-z 穿越 tau_off=4 的边界。

D5 是不重训、不修改 HDBSCAN 的最后一次因果收尾，不把 D4 重新解释为成功。
它在保留 semantic projection 的同时，将 frozen tree/vertical candidate 的
adapted offset-z 投影到 frozen base 的 tau_off 同一侧。XY residual 完全保留。
post-ensemble pipeline 必须同时满足：

- semantic membership mismatches = 0；
- full seed-set mismatches = 0；
- clustering seed 数严格等于 A0 的 325,596。

依次运行 tests.test_height_seed_identity、D5 L1W pipeline 和 D5 L1W evaluation。
对应配置为 pipeline_l1w_d5_full_seed_identity_hsca.yaml 与
evaluate_l1w_d5_full_seed_identity_hsca.yaml。

D5 Gate 仍固定为 Completeness 100%、F1 至少 98.4%、Commission 不高于 3.1%。
若失败，正式关闭 height-adapter 部署路线；若通过，先做 L1W exact seed-set
审计，再决定是否允许一次锁定 Wytham evaluation。不得扫描 projection margin。
## 11. D5 结果与路线关闭

D5 的机制断言全部通过：

- semantic membership mismatches = 0；
- full seed-set mismatches = 0；
- clustering seeds = 325,596，与 A0 完全一致；
- 13,104 个 protected offset-z 被投影回 frozen seed 边界同侧。

在只保留 HSCA XY vote 修正后，L1W 得到 Completeness 100.0%、Commission
3.7%、F1 98.1%、Precision 98.9%、Recall 99.0%、Coverage 98.0%。相对 A0，
F1 下降 0.3 pp、Commission 增加 0.6 pp，未通过固定 D5 Gate。

固定因果结论如下：

1. D2-A2 的 L1W 增益依赖会改变 seed 身份的 semantic/offset-z residual；
2. D3 证明这种 seed expansion/removal 在 Wytham 发生跨域放大；
3. D4 锁定 semantic 后性能退化，且仍遗留约 4.0% seed-count drift；
4. D5 锁定完整 seed set 后，单独保留的 attention XY vote 仍不优于 A0。

因此正式关闭 height-adapter 部署路线，不创建或运行 D5 Wytham 配置，不扫描
projection margin、checkpoint、attention 深度或 grouping 参数。HSCA 只作为论文中的
受控注意力消融与跨域失败分析，不作为最终部署模型。
