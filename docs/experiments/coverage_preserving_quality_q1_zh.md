# Q1 覆盖保护型实例质量 Oracle

## 目的

前序实验表明，提高 HDBSCAN `tau_min` 能降低 Commission，但会在 Wytham 中删除
真实的小树和林下树，使 Completeness 下降 3.066 个百分点。Q1 不再修改种子、vote、
聚类尺度或实例边界，而是验证能否同时学习两个不同风险：

```text
预测实例
  ├─ safe_reject：假阳性或冗余实例，可以删除
  └─ coverage_critical：某棵已检出 GT 树的唯一代表，必须保护
```

Q1 仅生成监督标签并计算固定 15% 最大拒绝预算下的 GT Oracle 上限。它不训练网络，
不读取 Wytham。

## 标签规则

- 只对 `target_classification_valid=True` 的实例生成二分类监督；
- 每个 `target_best_gt_id` 中 IoU 最高的正实例被标为 `coverage_critical`；
- IoU 并列时选择 instance ID 更小者，确保完全确定；
- classification-valid 假实例以及同一 GT 的其余冗余正实例标为 `safe_reject`；
- 边界实例、低标注覆盖实例、IoU 0.25～0.5 的模糊实例均为
  `protected_unknown`，本阶段不删除、不参与两个头的分类损失。

## 同步与测试

```bash
cd ~/projects/zrx/code/TreeLearn
git switch vertical-instance-quality
git pull --ff-only origin vertical-instance-quality

conda activate TreeLearn

python -m unittest \
  tests.test_coverage_preserving_quality_oracle \
  tests.test_coverage_preserving_quality_oracle_integration \
  -v
```

## 前置检查

```bash
test -f data/instance_quality/manifest.csv
find data/instance_quality/train -name '*.npz' | wc -l
find data/instance_quality/validation -name '*.npz' | wc -l
```

应分别存在 13 和 5 个 artifact。

## 运行

```bash
mkdir -p logs/coverage_preserving_quality
set -o pipefail

python -u \
  tools/diagnostics/diagnose_coverage_preserving_quality_oracle.py \
  --config configs/experiments/coverage_preserving_quality/q1_coverage_oracle.yaml \
  2>&1 | tee logs/coverage_preserving_quality/q1_coverage_oracle_run.log
```

该实验只加载约 9400 个实例，通常数分钟内完成，不需要 GPU。

## 输出与判断

```bash
cat logs/coverage_preserving_quality/q1_coverage_oracle/summary.md
```

同时生成：

```text
logs/coverage_preserving_quality/q1_coverage_oracle/dual_risk_labels.csv
```

预注册 Gate：

- train safe-reject ≥ 1000；
- train coverage-critical ≥ 3000；
- validation safe-reject ≥ 300；
- validation coverage-critical ≥ 1000；
- validation Oracle F1 至少提高 5 个百分点；
- Commission 至少降低 5 个百分点；
- 相对 baseline matched trees 的 Completeness 不得下降；
- 5/5 validation forests 必须全部取得正 F1 增益。

- `PASS`：进入 Q2 参数匹配双头 MLP，先证明 Coverage head 有用，再考虑关系注意力。
- `FAIL`：停止覆盖保护网络，不在 Wytham 上调整标签或拒绝比例。
