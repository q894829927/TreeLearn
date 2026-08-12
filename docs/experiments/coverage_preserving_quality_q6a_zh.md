# Q6a：欠分割碎片受限合并 Oracle

## 目标

Q5b1 的 proposal 激活 precision 最高只有 1.02%，因此 Seed-Completion 路线按预注册 Gate 关闭。Q3 尚有 41 棵漏检树被归因为 `fragmentation`，理论检测 F1 上限约为 1.225 个百分点。Q6a 先判断：只利用预测实例间的几何和 base-vote 邻接关系，能否生成覆盖这些碎片树的安全合并候选。

本阶段不训练网络、不修改 TreeLearn 主干或 HDBSCAN，也不读取 Wytham。

## 两个上限

1. `fragment_union_ceiling`：合并同一 Q3 fragmentation GT 的全部显著片段，表示理想碎片联合上限。
2. `fragment_graph_oracle`：先仅用预测信息建立实例邻接图，再由 GT Oracle 从图连通分量中验收安全合并。这个结果衡量 GT-free 候选图的上限，不是可部署方法。

固定邻接条件：

- base-vote 中心距离不超过 3.0 m；
- 原始 XY 中心距离不超过 8.0 m；
- XY 包围盒间距不超过 1.5 m；
- 垂直区间间距不超过 3.0 m。

任何 Oracle 合并都必须恢复对应 fragmentation 树、不丢失已有 TP，并使同一进程的全局检测 F1 严格提高。

## Gate

- 精确复现 41 个 Q3 fragmentation 目标；
- 理想联合至少恢复 20 棵，F1 至少提升 0.70 pp；
- GT-free 图至少覆盖 25 棵目标树；
- 图 Oracle 至少恢复 10 棵，F1 至少提升 0.50 pp；
- 不丢失已有 TP，Commission 不增加；
- 至少 4/5 个验证森林 F1 非负。

通过后进入 Q6b，在固定 train/validation forests 生成图边数据，并训练候选排序/验收头。失败则关闭 fragmentation 合并路线，不得使用 Wytham 调参。

## 服务器执行

```bash
cd ~/projects/zrx/code/TreeLearn
git switch vertical-instance-quality
git pull --ff-only origin vertical-instance-quality

conda activate TreeLearn
mkdir -p logs/coverage_preserving_quality

python -m unittest \
  tests.test_fragment_graph_oracle \
  tests.test_omission_diagnostics -v
```

运行完整五森林诊断：

```bash
nohup python -u \
  tools/diagnostics/diagnose_fragment_graph_oracle.py \
  --config configs/experiments/coverage_preserving_quality/q6a_fragment_graph_oracle.yaml \
  > logs/coverage_preserving_quality/q6a_fragment_graph_oracle_run.log \
  2>&1 < /dev/null &

echo $! | tee \
  logs/coverage_preserving_quality/q6a_fragment_graph_oracle.pid

tail -f \
  logs/coverage_preserving_quality/q6a_fragment_graph_oracle_run.log
```

脚本会校验并跳过已经完成的森林；异常中断后直接重复相同命令，不要加 `--force`。每个森林的底层日志位于：

```text
logs/coverage_preserving_quality/q6a_fragment_graph_oracle/logs/<plot>.log
```

完成后查看：

```bash
cat logs/coverage_preserving_quality/q6a_fragment_graph_oracle/summary.md
```
