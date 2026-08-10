# E3a 复杂度感知多尺度分组 Oracle 实验

## 1. 为什么停止 E2b

E2b 在保持 0.6 m Cell purity 与 collision 完全不变的情况下，只修正了
4.825% 的树种子，Mean vote error 仅下降 0.856%，P90 不变，而且五个验证森林的
聚类 F1 全部下降。Macro F1 下降 2.039 个百分点，Commission 反而增加 2.940 个
百分点。

这说明 Cell 归属不变并不等价于 HDBSCAN 连通结构不变。只移动少量“安全点”仍会
改变 core distance、mutual-reachability graph 和最小生成树；同时，部分点得到修正、
其余点保持原位会把原本一致的 vote 模态撕开。E2a 的 Pointwise Oracle 近乎零误差却
降低 F1，也已经说明“锚点误差最小”不是实例聚类的充分目标。

因此 E2a/E2b 的 vote-refinement 与 seed-attention 结论保持 `FAIL`，不得修改 Gate，
不得去 Wytham 搜索阻尼系数。

## 2. 新的独立假设

下一步改为验证：复杂林分是否需要不同的聚类尺度，而不是更精确的单一 vote。

```text
冻结的 TreeLearn base votes
        ↓
固定五个 HDBSCAN min_cluster_size
  [25, 35, 50, 75, 100]
        ↓
多尺度实例 proposal pool
        ↓
GT-only proposal selector Oracle
        ↓
判断非原版尺度能否恢复 mcs=50 漏掉的树
```

本阶段不修改网络、不删除种子、不修改 votes。`50` 是原版尺度。GT 仅用于计算
proposal pool 的恢复上限，不能部署。

## 3. 主要决策指标

Oracle 的 F1 会因为可以丢弃所有 unmatched proposal 而偏乐观，所以它只作为报告项，
不作为核心 Gate。主 Gate 使用：

- 相对 `mcs=50` 的 Tree recall 增益至少 1.0 个百分点；
- 五个森林合计至少新恢复 15 棵 GT 树；
- 至少 3/5 森林存在严格新增恢复；
- 至少 5% 的 GT 树，其最佳 proposal 来自非原版尺度；
- proposal 数量相对原版的平均膨胀不超过 5.5 倍；
- `mcs=50` 的标签摘要与 E2a 必须完全一致。

这些条件已预注册，不得在看到结果后放宽。

## 4. 同步与测试

```bash
cd ~/projects/zrx/code/TreeLearn
git switch vertical-instance-quality
git pull --ff-only origin vertical-instance-quality

conda activate TreeLearn
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export PYTHONUTF8=1

python -m unittest \
  tests.test_multiscale_grouping_oracle \
  tests.test_multiscale_grouping_oracle_integration \
  -v
```

## 5. 前置检查

```bash
test -f data/seed_vote_refinement/manifest.csv
test -f logs/complexity_seed_attention/e2a_vote_refinement_oracle/summary.json

for plot in G4N G4W L1N O1N O1W; do
  test -f \
    logs/complexity_seed_attention/e2a_vote_refinement_oracle/plots/${plot}.json
done

echo "E3a prerequisites PASS"
```

## 6. 长时间运行

```bash
mkdir -p logs/complexity_seed_attention

nohup python -u \
  tools/diagnostics/diagnose_multiscale_grouping_oracle.py \
  --config configs/experiments/complexity_seed_attention/e3a_multiscale_grouping_oracle.yaml \
  > logs/complexity_seed_attention/e3a_multiscale_grouping_oracle_run.log \
  2>&1 < /dev/null &

echo $! | tee \
  logs/complexity_seed_attention/e3a_multiscale_grouping_oracle.pid

tail -f \
  logs/complexity_seed_attention/e3a_multiscale_grouping_oracle_run.log
```

每个森林、每个尺度都会保存独立 cluster cache。进程中断后直接运行同一命令即可续跑，
不要加 `--force`。五个森林共需 25 次 HDBSCAN；预计 CPU 运行约 4～8 小时。

检查进程：

```bash
pid=$(cat \
  logs/complexity_seed_attention/e3a_multiscale_grouping_oracle.pid)

ps -p "$pid" -o pid,%cpu,%mem,rss,etime,stat,cmd

tail -n 80 \
  logs/complexity_seed_attention/e3a_multiscale_grouping_oracle_run.log
```

## 7. 查看结果及后续

```bash
cat \
  logs/complexity_seed_attention/e3a_multiscale_grouping_oracle/summary.md
```

- `PASS`：进入 E3b，生成多尺度 proposal 特征；先训练参数匹配的 proposal MLP，随后
  训练单层关系注意力选择器。checkpoint 与规则只在固定 train/validation forests 上选择，
  Wytham 最后只运行一次。
- `FAIL`：关闭复杂度感知分组路线。保留已被跨域验证的实例质量排序与选择性风险控制
  作为论文主线，不再在 Wytham 上搜索聚类参数。

末尾的 `RuntimeError: E3a multi-scale proposal gate failed` 表示实验完成但科学 Gate
未通过，并非程序崩溃。
