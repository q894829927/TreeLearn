# E2a Cellwise Vote-Refinement Oracle 实验

## 1. 研究问题

E1d 已否定 Seed-Cell 均衡配额筛选：Cell Oracle 虽然相对同配额随机选择提高
6.69% utility，并将密度 CV 降低 30.61%，但只保留 Point Oracle 的 80.20%
utility，且非树比例从 Point Oracle 的 5.03% 上升到 17.14%。因此不再删除种子，
也不降低 E1d Gate。

E1d 同时发现 92.35% 的 utility 方差来自 Cell 之间。E2a 将这一信号用于一个新的、
更受约束的问题：一个 0.6 m vote cell 共享的 XY residual，是否足以在保留全部种子的
条件下改善 base-vote 几何和原版 HDBSCAN 聚类？

本阶段是 GT-only 上限诊断，不训练网络，不读取 Wytham，不改变 TreeLearn 的
`tau_min`、HDBSCAN 或候选种子定义。

## 2. 为什么必须生成新 artifact

旧 `data/seed_quality` 只保存 `target_vote_error_xy` 标量，无法恢复 XY 残差方向。
从误差长度猜测方向会产生错误 Oracle。代码因此新增精确字段：

- `target_base_vote_xy`；
- `target_vote_residual_xy`。

新数据单独写入 `data/seed_vote_refinement`，不会覆盖或修改旧 E1a 数据。只重新运行
固定的 G4N、G4W、L1N、O1N、O1W 五个 validation forests。

## 3. 固定模式

所有模式使用完全相同的候选种子：

1. `base`：原始 TreeLearn base vote；
2. `global_shared`：全森林共享一个 GT 中位残差；这是平移不变性控制，HDBSCAN
   结果必须与 base 相同；
3. `cellwise_shared`：按原始 base-vote XY 建立 0.6 m Cell，每个 Cell 只能使用一个
   GT 中位 XY residual，并对 Cell 内全部候选共同生效；
4. `pointwise_oracle`：每个树种子使用自己的精确残差，只作为不可部署理论上限。

Cell 内即使存在多棵树，也不能按 tree ID 分开修正。没有已标注树种子的 Cell 使用零
修正，防止 Oracle 获得不现实的自由度。

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
  tests.test_seed_quality_data \
  tests.test_cellwise_vote_refinement_oracle \
  tests.test_cellwise_vote_refinement_integration \
  tests.test_cellwise_vote_refinement_run \
  -v
```

## 5. 生成五个精确目标 artifact

建议使用后台命令。生成器支持断点式重启：已经完整生成的森林会显示 `SKIP`。

```bash
mkdir -p logs/complexity_seed_attention

nohup python -u tools/data_gen/gen_seed_quality_data.py \
  --config configs/experiments/complexity_seed_attention/e2a_generate_vote_refinement.yaml \
  --pilot \
  > logs/complexity_seed_attention/e2a_generate_vote_refinement.log \
  2>&1 < /dev/null &

echo $! | tee \
  logs/complexity_seed_attention/e2a_generate_vote_refinement.pid

tail -f \
  logs/complexity_seed_attention/e2a_generate_vote_refinement.log
```

五个 pipeline 一般需要约 1–3 小时，取决于磁盘和 GPU。完成后检查：

```bash
cat data/seed_vote_refinement/generation_summary.md
wc -l data/seed_vote_refinement/manifest.csv
```

manifest 应为 6 行：1 行表头和 5 个 validation forests。

## 6. 运行 E2a Oracle

E2a 对 base、cellwise 和 pointwise 三种 vote 依次运行原版 HDBSCAN。预计需要
1–3 小时，主要消耗 CPU 和内存。每个森林完成后立即缓存到
`logs/complexity_seed_attention/e2a_vote_refinement_oracle/plots/`；进程中断后直接重跑
相同命令即可从下一森林继续。

```bash
nohup python -u \
  tools/diagnostics/diagnose_cellwise_vote_refinement_oracle.py \
  --config configs/experiments/complexity_seed_attention/e2a_vote_refinement_oracle.yaml \
  > logs/complexity_seed_attention/e2a_vote_refinement_oracle_run.log \
  2>&1 < /dev/null &

echo $! | tee \
  logs/complexity_seed_attention/e2a_vote_refinement_oracle.pid

tail -f \
  logs/complexity_seed_attention/e2a_vote_refinement_oracle_run.log
```

只有在 artifact 或配置发生变化、确实需要全部重算时才加入 `--force`。

## 7. 查看结果

```bash
cat \
  logs/complexity_seed_attention/e2a_vote_refinement_oracle/summary.md

column -s, -t < \
  logs/complexity_seed_attention/e2a_vote_refinement_oracle/per_plot_metrics.csv | \
  less -S
```

末尾出现 `E2a vote-refinement gate failed` 表示诊断正常完成但上限不足，不是程序崩溃。

## 8. 预注册 Gate

必须同时满足：

- 五个固定 validation forests 全部完成；
- Pointwise Oracle mean error 不超过 0.00001 m，证明标签导出和计算一致；
- Cellwise mean error 相对 base 至少下降 10%；
- Cellwise P90 error 相对 base 至少下降 5%；
- Cellwise 至少保留 Pointwise Oracle 70% 的 mean-error 改善；
- Cell purity 不下降；
- multi-tree collision cell rate 不增加；
- 至少 4/5 森林同时取得 error 与聚类非负收益；
- seed-cluster proxy F1 至少提高 0.5 个百分点；
- seed-cluster tree recall 最多下降 0.5 个百分点。

不得在结果出来后修改 Cell size、HDBSCAN、IoU 阈值或 Gate，也不得使用 Wytham
选择参数。

## 9. PASS / STOP

- PASS：实现参数量匹配的 `Cell-MLP residual` 控制组和单层
  `Cell-Query residual`，共享相同 Cell token、数据、随机种子和零初始化 residual
  head；只在固定 validation forests 选择模型。
- STOP：关闭 Cellwise vote-refinement 和种子注意力路线，不在 Wytham 上搜索 Cell
  size。论文主线保留已经验证的实例质量排序与选择性风险控制结果。
