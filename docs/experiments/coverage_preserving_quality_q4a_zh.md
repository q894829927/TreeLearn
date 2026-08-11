# Q4a：拓扑感知实例拆分 Oracle

## 目的

Q3 在固定五个 validation forests 上发现 149 棵漏检树，其中 73 棵属于
`undersegmentation`。这些树平均有较高 recall、较低 precision，说明主要问题是多棵树被
合并进同一个预测实例。Q4a 先验证“拆分”是否具有足够上限，不训练网络，也不读取
Wytham。

## 固定对照

四种模式共享 Q3 checkpoint、TreeLearn predictions、评估阈值与父实例集合：

1. `gt_extraction_ceiling`：从父实例中精确提取漏检 GT 点，只表示绝对上限；
2. `raw_xy_kmeans_oracle`：已知子树数量，使用原始 XY；
3. `base_vote_kmeans_oracle`：已知子树数量，使用冻结 base-vote XY；
4. `vertical_axis_kmeans_oracle`：已知子树数量，联合 base-vote 与高度条件轴轨迹。

后三种方法的 K 和是否接受拆分由 GT Oracle 给出，因此仍然只是 proposal 上限。几何
特征本身不读取 GT。每个父实例的 K-Means 最多使用 50,000 个确定性等距采样点拟合，
之后为全部父实例点分配簇。

## 运行

```bash
cd ~/projects/zrx/code/TreeLearn
conda activate TreeLearn
mkdir -p logs/coverage_preserving_quality

nohup python -u tools/diagnostics/diagnose_instance_split_oracle.py \
  --config configs/experiments/coverage_preserving_quality/q4a_instance_split_oracle.yaml \
  > logs/coverage_preserving_quality/q4a_instance_split_oracle_run.log \
  2>&1 < /dev/null &

echo $! | tee \
  logs/coverage_preserving_quality/q4a_instance_split_oracle.pid

tail -f \
  logs/coverage_preserving_quality/q4a_instance_split_oracle_run.log
```

逐森林串行运行。完成的森林会校验 Q3 baseline 与欠分割计数后显示 `SKIP`，所以中断后
直接执行同一命令即可续跑。仅在确认产物损坏时使用 `--force`。

## 查看状态和结果

```bash
pid=$(cat logs/coverage_preserving_quality/q4a_instance_split_oracle.pid)
ps -p "$pid" -o pid,%cpu,%mem,rss,etime,stat,cmd
pgrep -af "[d]iagnose_instance_split_oracle.py"
pgrep -af "[r]un_instance_split_oracle_pipeline.py"

tail -n 100 \
  logs/coverage_preserving_quality/q4a_instance_split_oracle_run.log

cat logs/coverage_preserving_quality/q4a_instance_split_oracle/summary.md
```

每个森林的底层 pipeline 日志位于：

```text
logs/coverage_preserving_quality/q4a_instance_split_oracle/logs/<plot>.log
```

## 固定 Gate

- 精确拆分上限：F1 至少 `+1.0 pp`，Completeness 至少 `+2.0 pp`，且不丢失原 TP；
- 最佳几何 proposal：恢复不少于 22 棵欠分割树，或 F1 至少 `+0.70 pp`；
- Commission 最多恶化 `1.0 pp`，Completeness 最多下降 `0.5 pp`；
- 最佳几何 proposal 至少在 4/5 个森林上提高 F1；
- 只有 Vertical 模式是最佳几何模式，并且相对 Base-vote 至少提高 `0.30 pp` F1，或
  多恢复 8 棵树时，才进入垂直拓扑拆分注意力；
- 若主 Gate 通过但垂直 Gate 未通过，只实现简单几何拆分控制；
- 主 Gate 失败则关闭实例拆分网络路线。

Wytham 不得用于选择 K、融合权重、Gate 或拆分阈值。

## Q3 目标一致性与断点续跑

Q4a 只处理 Q3 `gt_trees.csv` 中 `category=undersegmentation` 的 GT ID。
这里的类别是 Q3 按 semantic、base seed、density、undersegmentation 顺序确定的
“首个失败阶段”，不能再用最终 recall 独立推断，否则会把早期阶段失败的树重复纳入。

每个 Q4a artifact 会保存并校验这组 GT ID。旧版本或中断生成的 artifact 会显示
`STALE <plot>`，随后只重建对应森林；有效 artifact 仍显示 `SKIP <plot>`。因此修复后
直接重复执行原命令即可，不需要手动删除结果目录，也不需要添加 `--force`。
