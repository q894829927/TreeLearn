# Q2 参数匹配 Coverage-CVaR 实例质量 MLP

## 研究问题

Q1 证明固定 15% 拒绝预算存在稳定的安全过滤上限。但 Q1 的
`safe_reject` 与 `coverage_critical` 在 classification-valid 集合内互补，直接训练两个
BCE head 只是在重复一个二分类目标，不能作为可靠创新。

Q2 因此验证更明确的假设：

> 对风险最高的关键真树施加 CVaR 尾部惩罚，能否在保持拒绝假实例能力的同时，减少
> 最容易被过滤器误删的小树、稀疏树或困难树？

本阶段仍不实现注意力。Coverage-CVaR 先优于同参数 MLP 后，Q3 才允许加入实例关系
注意力。

## 参数匹配对照

三个模型均使用：

```text
35D global instance features
        ↓
MLP 64 → 32
        ↓
两个线性输出层
```

- `quality_iou_control`
  - 输出 1：原始 true-tree quality；
  - 输出 2：IoU regression；
  - rejection risk = `1 - quality × predicted_iou`；
  - 仅作为原始监督参考。
- `safe_iou_control`
  - 输出 1：safe-reject probability；
  - 输出 2：IoU regression；
  - rejection risk = `safe`；
  - 这是 Coverage-CVaR 的主控制组。
- `coverage_cvar`
  - 网络、参数量、safe BCE 和 IoU loss 与 `safe_iou_control` 完全相同；
  - 对每个 batch 中风险最高的 20% coverage-critical 真树增加 CVaR 惩罚；
  - CVaR weight 固定为 0.5；
  - rejection risk 仍为 `safe`，避免推理阶段增加额外规则。

## 锁定设置

- train forests：Q1 固定 13 个森林；
- validation forests：G4N、G4W、L1N、O1N、O1W；
- Wytham：禁止读取；
- seeds：42、43、44；
- 每个森林分别拒绝风险最高的 15%；
- 风险并列时 instance ID 小者先拒绝；
- checkpoint：先满足 Completeness 下降不超过 1 个百分点，再选择 F1 最高 epoch；
- IoU loss weight：0.5；Coverage-CVaR weight：0.5；tail fraction：0.2；
- AdamW：学习率 `1e-3`，weight decay `1e-3`；
- 最多 100 epochs，patience 15。

## 同步与测试

```bash
cd ~/projects/zrx/code/TreeLearn
git switch vertical-instance-quality
git pull --ff-only origin vertical-instance-quality

conda activate TreeLearn

python -m unittest \
  tests.test_coverage_preserving_quality_training \
  tests.test_coverage_preserving_quality_config \
  tests.test_coverage_quality_model \
  -v
```

## 前置检查

```bash
test -f data/instance_quality/manifest.csv
test -f logs/coverage_preserving_quality/q1_coverage_oracle/summary.json
test -f logs/coverage_preserving_quality/q1_coverage_oracle/dual_risk_labels.csv
```

确认 Q1：

```bash
python - <<'PY'
import json

path = 'logs/coverage_preserving_quality/q1_coverage_oracle/summary.json'
with open(path, encoding='utf-8') as file:
    report = json.load(file)

print('Q1 passed:', report['gate']['passed'])
print('reject ratio:', report['settings']['reject_ratio'])
PY
```

必须得到 `True` 和 `0.15`。

## 训练

```bash
mkdir -p logs/coverage_preserving_quality

nohup env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  python -u tools/training/train_coverage_preserving_quality.py \
  --config configs/experiments/coverage_preserving_quality/q2_coverage_cvar_mlp.yaml \
  > logs/coverage_preserving_quality/q2_coverage_cvar_mlp_run.log \
  2>&1 < /dev/null &

echo $! | tee logs/coverage_preserving_quality/q2_coverage_cvar_mlp.pid
tail -f logs/coverage_preserving_quality/q2_coverage_cvar_mlp_run.log
```

数据量约 9400 个实例，应明显短于 TreeLearn 主干训练。

## 查看结果

```bash
cat logs/coverage_preserving_quality/q2_coverage_cvar_mlp/summary.md
cat logs/coverage_preserving_quality/q2_coverage_cvar_mlp/per_seed_metrics.csv
```

## 预注册 Gate

- 三个模型参数量完全相同；
- Coverage-CVaR 每个 seed 的 Completeness 下降均不超过 1.0 个百分点；
- Coverage-CVaR 相对未过滤 baseline 平均 F1 至少提高 1.0 个百分点；
- Commission 平均至少下降 2.0 个百分点；
- Coverage-CVaR 相对 `safe_iou_control` 平均 F1 至少提高 0.5 个百分点；
- Coverage-CVaR 相对 `safe_iou_control` 的 Completeness 最多下降 0.5 个百分点；
- 至少赢得 2/3 seeds；
- 至少赢得 4/5 validation forests；
- F1 的 seed sample standard deviation 不超过 1.0 个百分点。

- `PASS`：实现 Q3 参数匹配实例关系注意力，并锁定 Q2 loss 与 15% 比例；
- `STOP`：尾部覆盖保护目标没有优于强控制组，不实现关系注意力，不使用 Wytham 调参。
