# Q4b0：Raw-XY 实例拆分可部署性分解

## 目的

Q4a 证明 Raw-XY known-K 拆分在固定 validation forests 上具有 `+0.798 pp`
F1 上限，但同时使用了 GT 子树数量 K 和 GT 拆分验收。Q4b0 不训练网络，只依次
移除这两个 Oracle 条件，决定 Q4b1 最小需要哪些预测头。

候选父实例仍严格来自 Q3 `category=undersegmentation`，因此 Q4b0 本身仍不是可部署
系统，也不得在 Wytham 上选择模式或 Gate。

## 固定对照

1. `known_k_oracle_accept`：Q4a Raw-XY 原结果，必须逐森林精确复现；
2. `known_k_accept_all`：保留 GT K，但取消 GT 验收，所有成功 KMeans proposal 都接受；
3. `fixed_k2_accept_all`：所有候选父实例固定拆成两棵，并接受所有成功 proposal。

## Gate

两个 accept-all 模式分别使用同一组 Gate：

- 汇总 F1 至少提升 `0.5 pp`；
- Completeness 下降不超过 `0.2 pp`；
- Commission 最多恶化 `0.5 pp`；
- 至少 `4/5` 个验证森林 F1 不下降。

决策规则：

- fixed-K2 通过：Q4b1 只训练欠分割候选分类头；
- 仅 known-K 通过：Q4b1 训练候选分类头与子树数量 K 预测头；
- 两者均未通过：Q4b1 还必须增加安全验收头。

## 运行

```bash
cd ~/projects/zrx/code/TreeLearn
conda activate TreeLearn
mkdir -p logs/coverage_preserving_quality

nohup python -u \
  tools/diagnostics/diagnose_instance_split_deployability.py \
  --config configs/experiments/coverage_preserving_quality/q4b0_instance_split_deployability.yaml \
  > logs/coverage_preserving_quality/q4b0_instance_split_deployability_run.log \
  2>&1 < /dev/null &

echo $! | tee \
  logs/coverage_preserving_quality/q4b0_instance_split_deployability.pid

tail -f \
  logs/coverage_preserving_quality/q4b0_instance_split_deployability_run.log
```

逐森林底层日志位于：

```text
logs/coverage_preserving_quality/q4b0_instance_split_deployability/logs/<plot>.log
```

查看最终结果：

```bash
cat \
  logs/coverage_preserving_quality/q4b0_instance_split_deployability/summary.md
```

脚本支持断点续跑：有效 artifact 显示 `SKIP`，旧版或不完整 artifact 显示 `STALE`
并只重建对应森林。除非明确确认 artifact 损坏，否则不要使用 `--force`。
