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

## E1b：冻结主干的双头 Seed-MLP 控制组

E1a Full Gate 已通过，固定数据统计为：18 个森林、4,537,796 个训练候选、1,908,145 个验证候选、252,638 个训练 coverage-critical targets，已知标签率 100%。E1b 不再运行 TreeLearn，也不读取点云；只读取 E1a 固定 artifact。

网络与后续 Attention 使用完全相同的输入和输出：

~~~text
32D frozen backbone + 29D GT-free scalar features
                    ↓
           MLP 61 → 128 → 64
       LayerNorm + GELU + Dropout
              ↙             ↘
  Reliability/Utility       Coverage
~~~

损失为：

~~~text
BCE(reliability) + 0.5 × SmoothL1(sigmoid(reliability), utility)
                 + 1.0 × BCE(coverage-critical)
~~~

训练每轮从 13 个训练森林各抽取 65,536 个候选，避免大森林主导。验证使用全部 5 个固定 validation forests。三次训练种子固定为 42/43/44，部署 checkpoint 预先锁定 seed 42，不根据结果挑最好的 seed。

选择规则同样预先固定，并在每个森林内独立执行：

1. 总共精确保留 `ceil(N × 0.78)` 个候选；
2. 先保护 Coverage 分数最高的 `ceil(N × 0.10)` 个候选；
3. 剩余名额按 Reliability 分数填充；
4. 与 Reliability-only、三个 exact-random controls 和 target Oracle 同时比较。

### 服务器运行

~~~bash
cd ~/projects/zrx/code/TreeLearn
git switch vertical-instance-quality
git pull --ff-only origin vertical-instance-quality

conda activate TreeLearn
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export PYTHONUTF8=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p logs/complexity_seed_attention

python -m unittest \
  tests.test_seed_quality_data \
  tests.test_seed_quality_mlp \
  -v
~~~

测试全部 `OK` 后启动三 seed 训练：

~~~bash
nohup env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  python -u tools/training/train_seed_quality_mlp.py \
  --config configs/experiments/complexity_seed_attention/e1b_train_seed_mlp.yaml \
  > logs/complexity_seed_attention/e1b_seed_mlp_run.log \
  2>&1 < /dev/null &

echo $! | tee logs/complexity_seed_attention/e1b_seed_mlp.pid
tail -f logs/complexity_seed_attention/e1b_seed_mlp_run.log
~~~

A6000/4090 预计约 1–3 小时，主要受 NPZ 解压、CPU→GPU 传输和全量验证排序影响。日志每次验证都会打印：

~~~text
[seed 42] epoch ... loss=... rel_AP=... cov_AP=...
critical_recall=... critical_tree_coverage=...
~~~

另开终端监控：

~~~bash
pid=$(cat logs/complexity_seed_attention/e1b_seed_mlp.pid)
ps -p "$pid" -o pid,%cpu,%mem,rss,etime,stat,cmd
nvidia-smi

tail -n 30 logs/complexity_seed_attention/e1b_seed_mlp_run.log
~~~

判断是否结束：

~~~bash
pid=$(cat logs/complexity_seed_attention/e1b_seed_mlp.pid)
if ps -p "$pid" > /dev/null; then
  echo "E1b still running"
else
  echo "E1b finished"
fi
~~~

完成后查看：

~~~bash
cat logs/complexity_seed_attention/e1b_seed_mlp/summary.md
column -s, -t < \
  logs/complexity_seed_attention/e1b_seed_mlp/per_seed_metrics.csv | less -S
ls -lh logs/complexity_seed_attention/e1b_seed_mlp/checkpoints
~~~

固定产物包括：

- `checkpoints/seed_mlp_seed42.pth`、`seed_mlp_seed43.pth`、`seed_mlp_seed44.pth`；
- `locked_validation_scores.npz`：只保存预先锁定的 seed 42 验证分数；
- `summary.json`、`summary.md`、`per_seed_metrics.csv`。

### E1b Gate 与下一步

E1b 必须同时满足：Reliability ROC-AUC ≥ 0.80；Coverage AP 至少为正类比例的 2 倍；双头 Critical recall 相对随机至少提高 0.05、相对 Reliability-only 至少提高 0.02；selected utility 相对随机至少提高 3%；所有 seed 的 critical-tree coverage ≥ 98%，每个验证森林 ≥ 95%；Critical recall 的 seed 标准差 ≤ 0.02。

- PASS：下一步实现同输入、同双头损失、同三 seed 和同选择器的 Complexity Seed Attention，并要求稳定优于该 MLP；
- `coverage_head_gain_passed=False`：Coverage target 对逐点 MLP 不可学，先修改覆盖标签或增加显式邻域关系；
- `reliability_auc_passed=False`：当前无标签特征不足，先检查 target/feature 对齐；
- 其他 Gate FAIL：保留所有 checkpoint 和报告用于诊断，但不进入 pipeline，也不读取 Wytham。

脚本末尾若因 Gate FAIL 抛出 `RuntimeError`，但 `summary.md` 已生成，这表示实验正常结束且科学判据未通过，不是训练程序崩溃。

## E1b2：Reliability 瓶颈审计

E1b 的 Coverage 分支、关键种子召回、utility 增益、树覆盖率和三 seed 稳定性均通过，但固定的 Reliability ROC-AUC Gate 未通过（0.733786 < 0.80）。不得事后降低 Gate，也不得直接查看 Wytham 或进入 pipeline。E1b2 只读取固定 E1a validation artifacts、seed 42 的锁定验证分数以及 E1b 报告。

E1b2 分解以下问题：

1. artifact、candidate index、标签与锁定验证分数是否逐元素对齐；
2. Reliability 分数是在区分树/非树时失败，还是在树点内部区分高低质量 vote 时失败；
3. global AUC 与每森林 macro AUC 的差距是否表明跨森林校准问题；
4. 原 0.50 utility 阈值附近是否存在大量边界样本；
5. 仅作诊断的 utility 阈值扫描是否显著改变 AUC；
6. 32D backbone 与 29D scalar 中最强单变量信号是否已经足够强；
7. score 与连续 utility 的排序相关性。

同步代码并先运行测试：

~~~bash
cd ~/projects/zrx/code/TreeLearn
git switch vertical-instance-quality
git pull --ff-only origin vertical-instance-quality

conda activate TreeLearn
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export PYTHONUTF8=1
mkdir -p logs/complexity_seed_attention

python -m unittest \
  tests.test_seed_quality_mlp \
  tests.test_seed_reliability_bottleneck \
  -v
~~~

测试通过后运行审计：

~~~bash
python -u tools/diagnostics/diagnose_seed_reliability_bottleneck.py \
  --config configs/experiments/complexity_seed_attention/e1b2_reliability_bottleneck.yaml \
  2>&1 | tee logs/complexity_seed_attention/e1b2_reliability_bottleneck_run.log
~~~

本阶段只读取 5 个验证森林，预计运行数分钟；61 个单变量特征的排序统计是主要耗时。结果位于：

~~~bash
cat logs/complexity_seed_attention/e1b2_reliability_bottleneck/summary.md
~~~

自动建议的处理规则在运行前固定：

- `repair_alignment`：数据或锁定分数没有逐元素对齐，先修复；
- `forest_calibrated_ranking`：每森林 AUC 明显高于 global AUC，先处理域校准；
- `continuous_utility_ranking`：二值阈值敏感或边界样本过多，改用连续 utility 排序损失；
- `capacity_optimization_probe`：单变量已有很强信号但 MLP 没学到，先排查优化；
- `parameter_matched_capacity_probe`：运行更强但仍为逐点的参数匹配控制；
- `relational_attention_candidate`：标签稳定、跨森林差距小且树内逐点信号弱，才实现显式邻域 Complexity Seed Attention。

阈值扫描只用于确定问题类型，不会据此修改 E1b 标签或在 Wytham 上调参。

## E1c：参数匹配的 Complexity Seed Relation Attention

E1b2 自动建议为 `relational_attention_candidate`。锁定诊断显示：tree-vs-non-tree AUC 为 0.937574，但 within-tree reliability AUC 仅为 0.659592；utility 阈值最大诊断增益仅 0.009237、0.50±0.10 边界率仅 14.360%、macro−global 为 -0.028248，说明主要瓶颈不是标签边界或跨森林校准，而是逐点/聚合特征缺少候选种子之间的显式关系。

E1c 仍只使用固定 13 个训练森林和 5 个验证森林，不读取 Wytham，也不修改 E1a 标签、E1b selector 或原始 TreeLearn。邻域在 predicted base-vote XY 空间内构建：`K=8`、最大半径 `0.6 m`、第一列始终为 query 自身、邻居不得跨森林。

为了判断增益是否真的来自注意力，同时训练两个模型：

1. `Neighborhood-MLP`：query 61D、邻域 61D 均值/标准差、6D 相对几何均值/标准差；隐藏维度 176→96；
2. `Relation-Attention`：61D→64D，单层向量注意力，relative base-vote XY、vote distance、point XY 和 Z 共同形成 6D 位置编码，随后使用 128D FFN。

两者参数量预期约为 52.2K 与 50.4K，差异必须小于 10%；使用完全相同的邻接缓存、训练采样、Reliability/Coverage 双头损失、seed 42/43/44 和 78%/10% selector。

### 服务器同步与测试

~~~bash
cd ~/projects/zrx/code/TreeLearn
git switch vertical-instance-quality
git pull --ff-only origin vertical-instance-quality

conda activate TreeLearn
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export PYTHONUTF8=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p logs/complexity_seed_attention

python - <<'PY'
from scipy.spatial import cKDTree
import torch
print('SciPy cKDTree: OK')
print('PyTorch:', torch.__version__)
print('CUDA:', torch.cuda.is_available())
PY

python -m unittest \
  tests.test_seed_attention_neighbors \
  tests.test_seed_relation_attention \
  tests.test_seed_relation_training_smoke \
  -v
~~~

### E1c-a：生成固定邻域缓存

~~~bash
nohup python -u tools/data_gen/gen_seed_attention_neighbors.py \
  --config configs/experiments/complexity_seed_attention/e1c_generate_seed_neighbors.yaml \
  > logs/complexity_seed_attention/e1c_generate_neighbors.log \
  2>&1 < /dev/null &

echo $! | tee logs/complexity_seed_attention/e1c_generate_neighbors.pid
tail -f logs/complexity_seed_attention/e1c_generate_neighbors.log
~~~

缓存生成支持断点续跑：已经存在的森林会重新验证后 `SKIP`。不要使用 `--force`，除非确认缓存损坏。预计约 20–60 分钟，输出约数百 MB。

完成后检查：

~~~bash
cat data/seed_quality_neighbors/summary.md
wc -l data/seed_quality_neighbors/manifest.csv
du -sh data/seed_quality_neighbors
~~~

必须显示所有 Gate 为 `True` 和最终 `PASS`，才可以启动训练。

### E1c-b：训练参数匹配对照

~~~bash
nohup env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  python -u tools/training/train_seed_relation_attention.py \
  --config configs/experiments/complexity_seed_attention/e1c_train_seed_relation_attention.yaml \
  > logs/complexity_seed_attention/e1c_seed_relation_attention_run.log \
  2>&1 < /dev/null &

echo $! | tee logs/complexity_seed_attention/e1c_seed_relation_attention.pid
tail -f logs/complexity_seed_attention/e1c_seed_relation_attention_run.log
~~~

A6000/4090 预计约 4–10 小时。脚本先打印两种模型参数量，随后依次运行 Neighborhood-MLP 的三个 seed 和 Relation-Attention 的三个 seed。监控命令：

~~~bash
pid=$(cat logs/complexity_seed_attention/e1c_seed_relation_attention.pid)
ps -p "$pid" -o pid,%cpu,%mem,rss,etime,stat,cmd
nvidia-smi
tail -n 40 logs/complexity_seed_attention/e1c_seed_relation_attention_run.log
~~~

最终结果：

~~~bash
cat logs/complexity_seed_attention/e1c_seed_relation_attention/summary.md
column -s, -t < \
  logs/complexity_seed_attention/e1c_seed_relation_attention/per_seed_metrics.csv | less -S
~~~

E1c Gate 预先固定为：参数差不超过 10%；Attention 平均 overall Reliability AUC≥0.80、tree-only AUC≥0.70；相对 Neighborhood-MLP 的 overall AUC 增益≥0.02、tree-only AUC 增益≥0.03；Coverage AP 和 Critical recall 分别最多下降 0.005；selected utility 相对提高至少 1%；tree-only AUC 至少赢得 2/3 seeds，且三 seed 标准差不超过 0.02。

- PASS：下一阶段才把锁定 seed 42 的 Relation-Attention 接入 L1W seed selector；
- FAIL：保留负结果，不进入 pipeline，也不在 Wytham 上修改邻域半径、K 或 Gate。
