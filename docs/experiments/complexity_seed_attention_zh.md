# 复杂度感知种子注意力实验

## E0：GT Seed Utility Oracle

本阶段只判断“学习种子可靠性是否存在足够上限”，不训练 MLP、Point Transformer 或任何
新模块，也不读取 Wytham。所有模式复用一次 TreeLearn pointwise inference 和 NumPy
ensemble，保证差异只来自初始聚类种子。

固定比较：

1. `baseline`：保留全部基础种子；
2. `random_s42/s43/s44`：随机保留 78%；
3. `oracle_global`：按 GT vote accuracy 与 vote-cell purity 的乘积保留全局最高 78%；
4. `oracle_balanced`：先在每棵 GT 树内部对 utility 排序，再全局保留 78%，模拟后续空间与
   高度平衡选择能够达到的理论上限。

Oracle utility 为：

~~~text
exp(-0.5 * (base_vote_xy_error / 0.30 m)^2)
× vote_cell_purity(0.60 m)
~~~

这是严格的离线上限，使用了 GT offset 与 instance label，不得用于正式 pipeline 或作为方法
结果。只有 `oracle_balanced` 同时提高 F1、降低 Commission、保持 Completeness，并超过三个
随机对照均值，才进入 E1。

## 服务器运行

~~~bash
cd ~/projects/zrx/code/TreeLearn
git switch vertical-instance-quality
git pull --ff-only origin vertical-instance-quality

conda activate TreeLearn
mkdir -p logs/complexity_seed_attention

nohup python -u tools/diagnostics/diagnose_complexity_seed_oracle.py \
  --config configs/experiments/complexity_seed_attention/e0_seed_oracle_l1w.yaml \
  > logs/complexity_seed_attention/e0_seed_oracle_l1w_run.log 2>&1 < /dev/null &

echo $! | tee logs/complexity_seed_attention/e0_seed_oracle_l1w.pid
tail -f logs/complexity_seed_attention/e0_seed_oracle_l1w_run.log
~~~

运行会依次打印一次推理和六次聚类的进度。HDBSCAN 阶段可能数分钟没有新日志，应通过 PID、
CPU 和日志时间共同判断，不要仅凭 `tail -f` 无输出终止。

~~~bash
pid=$(cat logs/complexity_seed_attention/e0_seed_oracle_l1w.pid)
ps -p "$pid" -o pid,%cpu,%mem,rss,etime,stat,cmd
cat logs/complexity_seed_attention/e0_seed_oracle_l1w/summary.md
~~~

## E0 Gate

- F1 相对 baseline 至少提高 `0.50 pp`；
- Commission 至少降低 `1.00 pp`；
- Completeness 下降不超过 `0.50 pp`；
- F1 至少高于三个 random controls 的均值 `0.25 pp`。

若脚本最后抛出 `Seed Oracle gate failed`，表示诊断正常完成但上限不足，应停止本路线；不得
通过修改 Wytham 参数寻找正结果。若 PASS，E1 才实现完全相同输入与标签的 MLP 控制组，随后
实现单层 Point Transformer；Attention 必须相对 MLP 提高 seed-ranking 指标后才能进入完整
pipeline 实验。

## E0b：未标注保护与空间约束 Oracle

E0 将 `instance_labels <= 0` 都视为非树，但部分标注的 L1W 实际使用
`-1=未标注`、`0=已知非树`、`>0=树实例`。E0b 因此固定比较：

1. `baseline`：全部基础种子，必须精确复刻 `TP=156, FP=5, FN=0`；
2. `oracle_masked`：只在已标注区域计算 utility，强制保护所有未标注候选；
3. `oracle_spatial`：额外为每棵树的每个 0.60 m vote 网格保留一个代表，并保证每树至少
   50 个种子（候选不足时全部保留）；
4. `random_spatial_s42/s43/s44`：共享相同保护与空间必选集合，只随机填充剩余名额。

所有模式严格保留 `ceil(N × 0.78)` 个候选；若必选集合超过预算，脚本在聚类前直接失败。

### 服务器运行

~~~bash
cd ~/projects/zrx/code/TreeLearn
git switch vertical-instance-quality
git pull --ff-only origin vertical-instance-quality

conda activate TreeLearn
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export PYTHONUTF8=1
mkdir -p logs/complexity_seed_attention

python -m unittest tests.test_complexity_seed_oracle_e0b -v

nohup python -u tools/diagnostics/diagnose_complexity_seed_oracle_e0b.py \
  --config configs/experiments/complexity_seed_attention/e0b_seed_oracle_l1w.yaml \
  > logs/complexity_seed_attention/e0b_seed_oracle_l1w_run.log 2>&1 < /dev/null &

echo $! | tee logs/complexity_seed_attention/e0b_seed_oracle_l1w.pid
tail -f logs/complexity_seed_attention/e0b_seed_oracle_l1w_run.log
~~~

完成后查看：

~~~bash
cat logs/complexity_seed_attention/e0b_seed_oracle_l1w/summary.md
cat logs/complexity_seed_attention/e0b_seed_oracle_l1w/metrics.csv
~~~

结果目录还会保存 `summary.json` 与 `selected_indices.npz`。

### E0b Gate 与下一步

Gate 不变：F1 提升至少 `0.50 pp`、Commission 降低至少 `1.00 pp`、Completeness 下降
不超过 `0.50 pp`，且空间 Oracle 的 F1 至少超过结构化随机均值 `0.25 pp`。

- PASS：下一步实现冻结 TreeLearn 主干的 Complexity Seed MLP 控制组；
- FAIL：关闭种子删减路线；
- E0b 只使用 L1W，不读取 Wytham，也不得根据 Wytham 调参。
## E1a：冻结主干的种子可靠性与覆盖目标数据

E0b 证明了空间约束 Oracle 有效，但单纯按可靠性排序会误删少量决定树木覆盖的关键种子。因此 E1 不直接训练注意力网络，先生成可审计、无输入泄漏的固定数据集。

每个原始 TreeLearn base-seed 候选保存两类输入：

1. 冻结官方 small-tree checkpoint 得到的 `32D backbone_features`；
2. `29D scalar_features`：5 个逐点量（树概率、verticality、归一化高度、XY offset 幅值、Z offset 幅值）以及在 `0.3/0.6/1.2 m` 三种 predicted base-vote 网格中的 24 个局部复杂度统计量。

GT 只进入以下监督字段，不进入输入字段：

- `target_reliable`：由 base-vote XY 误差与 0.6 m vote-cell purity 联合定义；
- `target_coverage_critical`：每棵树的每个 0.6 m vote cell 至少保留一个代表，同时每棵树至少保留 50 个候选（不足时全部保留）；
- 连续诊断目标：`target_vote_error_xy`、`target_vote_cell_purity`、`target_utility`。

固定森林划分为：

- train：A1N、A1W、G1N、G1W、G2N、G2W、G3N、G3W、L2N、L2W、LG1、LG2、LG3；
- validation：G4N、G4W、L1N、O1N、O1W；
- Wytham：本阶段禁止读取，后续也不能用于选模型或阈值。

### 第一步：服务器同步与测试

~~~bash
cd ~/projects/zrx/code/TreeLearn
git switch vertical-instance-quality
git pull --ff-only origin vertical-instance-quality

conda activate TreeLearn
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export PYTHONUTF8=1
mkdir -p logs/complexity_seed_attention

python -m unittest tests.test_seed_quality_data -v
~~~

只有测试全部 `OK` 才运行数据生成。

### 第二步：两森林 pilot

pilot 只运行 A1N（train）和 G4N（validation），用于确认显存、路径、artifact 维度与提前跳过 HDBSCAN 的逻辑。

~~~bash
nohup python -u tools/data_gen/gen_seed_quality_data.py \
  --config configs/experiments/complexity_seed_attention/e1a_generate_seed_quality_data.yaml \
  --pilot \
  > logs/complexity_seed_attention/e1a_pilot.log 2>&1 < /dev/null &

echo $! | tee logs/complexity_seed_attention/e1a_pilot.pid
tail -f logs/complexity_seed_attention/e1a_pilot.log
~~~

另开终端监控：

~~~bash
pid=$(cat logs/complexity_seed_attention/e1a_pilot.pid)
ps -p "$pid" -o pid,%cpu,%mem,rss,etime,stat,cmd
pgrep -af "[t]ools/pipeline/pipeline.py"
tail -f data/seed_quality/logs/A1N.log
~~~

pilot 完成后检查：

~~~bash
cat data/seed_quality/generation_summary.md
head -n 6 data/seed_quality/manifest.csv
head -n 6 data/seed_quality/audit_sample.csv
~~~

pilot 必须满足：两个 artifact 非空、backbone 维度为 32、scalar 维度为 29、所有出现于候选集的监督树均有 coverage target、无森林/组泄漏、审计样本不少于 20。任一失败都先修复，不进入 full。

### 第三步：完整生成

pilot PASS 后执行。已有且校验通过的 A1N/G4N 会自动 `SKIP`，因此不需要 `--force`。

~~~bash
nohup python -u tools/data_gen/gen_seed_quality_data.py \
  --config configs/experiments/complexity_seed_attention/e1a_generate_seed_quality_data.yaml \
  > logs/complexity_seed_attention/e1a_full.log 2>&1 < /dev/null &

echo $! | tee logs/complexity_seed_attention/e1a_full.pid
tail -f logs/complexity_seed_attention/e1a_full.log
~~~

运行中可查看当前子任务和单森林日志：

~~~bash
pid=$(cat logs/complexity_seed_attention/e1a_full.pid)
ps -p "$pid" -o pid,%cpu,%mem,rss,etime,stat,cmd
pgrep -af "[t]ools/pipeline/pipeline.py"
tail -n 30 logs/complexity_seed_attention/e1a_full.log
ls -lh data/seed_quality/train data/seed_quality/validation
~~~

只有在某个 artifact 已损坏且明确需要重建时，才对指定森林使用：

~~~bash
python -u tools/data_gen/gen_seed_quality_data.py \
  --config configs/experiments/complexity_seed_attention/e1a_generate_seed_quality_data.yaml \
  --plots A1N \
  --force
~~~

不要在完整运行尚未结束时启动第二份生成进程。

### E1a Full Gate 与后续分支

完整结果位于 `data/seed_quality/`，重点查看：

~~~bash
cat data/seed_quality/generation_summary.md
wc -l data/seed_quality/manifest.csv
~~~

Full Gate 要求：固定 18 个森林全部存在；train candidates 至少 500,000；validation candidates 至少 200,000；train coverage-critical 至少 10,000；已知非树候选至少 1,000；已知标签率至少 99.9%；critical 比例在 0.5%–50% 之间；维度、树覆盖、森林划分和审计样本全部通过。

- PASS：下一阶段 E1b 训练冻结主干的双头 MLP 控制组（Reliability + Coverage），只按固定 validation forests 选 checkpoint；
- 数据 Gate FAIL：修复数据或阈值定义后重新生成对应森林，不得通过查看 Wytham 来改 Gate；
- E1b MLP 有效后才实现 Complexity Seed Attention，并要求相对同输入 MLP 有稳定增益。
