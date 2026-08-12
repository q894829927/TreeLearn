# Q5b0：GT-free Coverage-Seed Proposal Oracle

## 1. Q5a 结论

Q5a 在固定五个 validation forests 上通过：预测量驱动的 `margin_topup`
恢复 13/31 棵 base-seed support failure 树，没有丢失基线树，Detection F1
提高 0.511 pp，并且 Commission 略有下降。Q5a 证明“补足到 `tau_min=50`”
有效，但仍由 GT 指定需要补种的目标树，不能直接部署。

Q5b0 只回答一个问题：能否在不知道目标树 ID 的情况下，先由预测量产生覆盖这些目标的
候选区域。只有候选生成上限通过，才值得训练 Q5b1 激活头。

## 2. GT-free proposal 定义

候选生成只读取冻结 pipeline 的预测量：semantic tree probability、base XY vote、
verticality、offset-z 和原始 base-seed mask。

在 base-vote 空间使用固定的 `0.6 m` 和 `1.2 m` 网格，并在 x/y 方向使用
`0` 与 `0.5` 个网格的固定平移，共八个视图。若一个 cell 至少包含 50 个 semantic
tree points，但原始 base seeds 少于 50，则成为 GT-free proposal。proposal 内仍使用
Q5a 已通过的 margin 排序，只补足到 50 个种子，不修改 vote 或 HDBSCAN 参数。

GT 只在评估端选择属于 Q3 固定 31 棵目标树的 proposal，然后完整重跑 HDBSCAN。
因此 Q5b0 是 proposal 召回上限，不是可部署系统，也不允许读取 Wytham。

## 3. 预注册 Gate

- proposal 覆盖至少 20/31 棵目标树；
- 最终恢复至少 10/31 棵；
- Detection F1 至少提高 0.35 pp；
- 最多损失 2 棵基线已检测树；
- Commission 最多增加 0.50 pp；
- 至少 4/5 个 validation forests 的 F1 不下降。

通过后进入 Q5b1：使用 13 个 train forests 生成相同 proposal，并训练小型激活头；
失败则关闭 Seed-Completion 路线。

## 4. 运行

```bash
conda activate TreeLearn
mkdir -p logs/coverage_preserving_quality

nohup python -u \
  tools/diagnostics/diagnose_seed_completion_proposal_oracle.py \
  --config configs/experiments/coverage_preserving_quality/q5b0_seed_completion_proposal_oracle.yaml \
  > logs/coverage_preserving_quality/q5b0_seed_completion_proposal_oracle_run.log \
  2>&1 < /dev/null &

echo $! | tee \
  logs/coverage_preserving_quality/q5b0_seed_completion_proposal_oracle.pid

tail -f \
  logs/coverage_preserving_quality/q5b0_seed_completion_proposal_oracle_run.log
```

查看单森林进度：

```bash
tail -f \
  logs/coverage_preserving_quality/q5b0_seed_completion_proposal_oracle/logs/G4N.log
```

完成后：

```bash
cat \
  logs/coverage_preserving_quality/q5b0_seed_completion_proposal_oracle/summary.md
```

Gate 失败时脚本会先保存完整报告，再以 `RuntimeError` 退出；这表示科学假设失败，
不是程序崩溃。已验证 artifact 支持自动 `SKIP`，不要随意添加 `--force`。
