# E2b 拓扑安全门控 Vote-Refinement Oracle 实验

## 1. E2a 结论保持不变

E2a 的无条件 Cellwise residual 在固定五个 validation forests 上取得：

- Mean XY error 下降 85.307%；
- P90 XY error 下降 87.756%；
- seed-cluster proxy F1 提升 1.208 个百分点；
- Commission 降低 1.463 个百分点；
- 4/5 森林取得非负聚类收益。

但 Cell purity 下降 0.538 个百分点，多树碰撞率增加 1.864 个百分点，且
O1N 的 F1 下降 1.146 个百分点。因此 E2a 必须保持 `FAIL`，不得降低原 Gate，
不得在 Wytham 上调整 Cell size 或 HDBSCAN。

Pointwise Oracle 将 vote error 几乎降为零，却使 Macro F1 从 79.577% 降至
76.981%。这说明逐点锚点误差不是充分目标：过度收缩可能让相邻树锚点发生合并。

## 2. 新的 E2b 假设

E2b 不重新解释 E2a，而是注册一个新的安全门控问题：

```text
原始 base votes
        ↓
固定 0.6 m Cell 共享 residual
        ↓
GT-only topology-safe binary gate
        ↓
v_safe = v_base + gate × residual_cell
        ↓
原版 HDBSCAN
```

一个源 Cell 只有同时满足以下条件才允许修正：

1. Cell 内全部候选属于同一棵已标注树；
2. 共享 residual 降低该 Cell 的 vote error；
3. 修正后的目标 Cell 为空，或只包含同一棵树；
4. 目标 Cell 中的非树点不会令一个新树 Cell 变得不纯；
5. 不同树的候选 Cell 不会被修正到同一个目标 Cell。
若上述初始门控仍因“树占据 Cell 数减少”导致 purity 或碰撞率比例变差，则启用
严格回退：只保留修正后所有候选仍停留在原 0.6 m 拓扑 Cell 的源 Cell。这样逐点
Cell 归属保持不变，两个拓扑指标在构造上不会退化。

该 gate 使用 GT，只是上限诊断，不能直接部署。它的作用是判断“残差 + 安全门控”
是否值得由后续 Cell-MLP 或 Cell-Query Attention 学习。

## 3. 固定对照

E2b 同时报告下列模式，但不根据 Wytham 选择阻尼：

- `base`；
- `damped_0p25`；
- `damped_0p50`；
- `damped_0p75`；
- `damped_1p00`，必须复现 E2a `cellwise_shared`；
- `topology_safe`，E2b 主决策模式；
- `pointwise_oracle`，不可部署理论上限。

Base、1.0 Cellwise 和 Pointwise 的 HDBSCAN 指标直接复用已验证的 E2a cache；
只新增三个阻尼模式和 topology-safe 模式的 HDBSCAN，从而减少重复计算。

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
  tests.test_cellwise_vote_refinement_oracle \
  tests.test_cellwise_vote_refinement_integration \
  tests.test_topology_safe_vote_refinement_oracle \
  tests.test_topology_safe_vote_refinement_integration \
  tests.test_topology_safe_vote_refinement_run \
  -v
```

## 5. 前置检查

E2b 不重新生成点云预测，但必须存在完整的 E2a artifact 和逐森林 cache：

```bash
test -f data/seed_vote_refinement/manifest.csv
test -f logs/complexity_seed_attention/e2a_vote_refinement_oracle/summary.json

for plot in G4N G4W L1N O1N O1W; do
  test -f \
    logs/complexity_seed_attention/e2a_vote_refinement_oracle/plots/${plot}.json
done
```

所有命令无输出且退出码为 0，才进入 E2b。

## 6. 运行 E2b

```bash
mkdir -p logs/complexity_seed_attention

nohup python -u \
  tools/diagnostics/diagnose_topology_safe_vote_refinement_oracle.py \
  --config configs/experiments/complexity_seed_attention/e2b_topology_safe_vote_oracle.yaml \
  > logs/complexity_seed_attention/e2b_topology_safe_vote_oracle_run.log \
  2>&1 < /dev/null &

echo $! | tee \
  logs/complexity_seed_attention/e2b_topology_safe_vote_oracle.pid

tail -f \
  logs/complexity_seed_attention/e2b_topology_safe_vote_oracle_run.log
```

三个阻尼模式和 topology-safe 模式各需要一次 HDBSCAN。预计总耗时约 2～5 小时，
主要使用 CPU 和内存。每个森林完成后写入独立 cache；进程中断后直接重跑同一命令，
不要加入 `--force`。

进程检查：

```bash
pid=$(cat \
  logs/complexity_seed_attention/e2b_topology_safe_vote_oracle.pid)

ps -p "$pid" \
  -o pid,%cpu,%mem,rss,etime,stat,cmd
```

## 7. 预注册 Gate

`topology_safe` 必须同时满足：

- 五个固定 validation forests 全部完成；
- Pointwise mean error 不超过 0.00001 m；
- Mean error 相对 base 至少下降 60%；
- P90 error 相对 base 至少下降 50%；
- Cell purity 最大下降不超过 0.1 个百分点；
- 多树碰撞率最大增加不超过 0.25 个百分点；
- seed-cluster F1 至少提高 1.0 个百分点；
- Commission 至少降低 1.0 个百分点；
- Tree recall 最大下降不超过 0.5 个百分点；
- 至少 4/5 森林 F1 非负提升；
- 任一森林 F1 不得下降超过 0.5 个百分点。

不得在看到结果后更改上述 Gate、Cell size、阻尼集合或 HDBSCAN 参数。

## 8. 查看结果及下一阶段

```bash
cat \
  logs/complexity_seed_attention/e2b_topology_safe_vote_oracle/summary.md
```

- `PASS`：实现参数量匹配的 `Cell-MLP Residual + Gate` 控制组，再实现单层
  `Cell-Query Attention Residual + Gate`；训练和 checkpoint 选择只使用固定
  train/validation forests，Wytham 保持锁定。
- `FAIL`：正式关闭 vote-refinement/seed-attention 路线，不在 Wytham 上继续搜索；
  论文主线保留已经跨域验证的实例质量排序与选择性风险控制。

末尾的 `RuntimeError: E2b topology-safe gate failed` 表示诊断完成但科学 Gate
未通过，不是程序崩溃。
