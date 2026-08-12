# Q5b1：无 GT Seed-Completion Proposal 激活头

## 1. 阶段目标

Q5b0 已证明：只使用预测 vote、semantic、verticality 和 offset 生成的 proposal，能够覆盖固定验证集的 31/31 棵 base-seed support failure 目标树；Oracle 激活 41 个 proposal 后恢复 12 棵树，F1 提升 0.362 个百分点且没有丢树。

Q5b1 不再使用 GT 生成候选。它只学习回答两个问题：

1. 该 proposal 是否应被激活，用于补足 HDBSCAN 的最小种子数；
2. 该 proposal 是否覆盖某棵 seed-support failure 目标树。

第二个任务是辅助覆盖头，用来减轻 Oracle 激活标签极稀疏、等价 proposal 被标成负例的问题。最终排序分数是两个头概率的几何平均。

本阶段只生成和训练 proposal 激活器，不修改 TreeLearn 主干、vote、HDBSCAN 参数，也不运行 Wytham。

## 2. 固定数据划分

训练森林共 13 个：

`A1N, A1W, G1N, G1W, G2N, G2W, G3N, G3W, L2N, L2W, LG1, LG2, LG3`

验证森林共 5 个：

`G4N, G4W, L1N, O1N, O1W`

代码会检查森林和同组林分泄漏，并拒绝任何包含 Wytham 的配置。

验证 proposal 直接导入已经通过 Gate 的 Q5b0 artifact，避免再次运行 HDBSCAN 造成计数漂移。训练 proposal 使用冻结的 TreeLearn checkpoint 在 13 个训练森林上生成。

## 3. 标签与特征

### 3.1 标签

- `activation_target`：Q5b0 greedy Oracle 为补足 `tau_min=50` 实际激活的 proposal。
- `coverage_target`：proposal 的新增种子中包含 seed-support failure 目标树点。
- `target_tree_ids`：proposal 覆盖的目标树 ID，只用于验证目标覆盖率。

### 3.2 输入特征

每个 proposal 使用 15 个无 GT 特征：

- 网格尺度和 XY shift；
- semantic 点数、已有 base seed 数、需补 seed 数；
- tree probability 的均值和标准差；
- verticality 的均值和标准差；
- offset-z 绝对值的均值和标准差；
- vote 半径；
- margin 的均值和标准差。

## 4. 比较模型

按相同训练/验证数据比较：

1. 固定几何规则；
2. 双目标 Logistic Regression；
3. 双头 MLP（主实验）。

MLP 只训练 proposal 激活分支，不更新 TreeLearn。训练 seed 固定为 `42, 43, 44`，部署 seed 固定为 `42`。

## 5. 预注册 Gate

模型必须同时满足：

- Activation AP / prevalence 至少 20 倍；
- Coverage AP / prevalence 至少 3 倍；
- 选中 proposal 的 activation precision 至少 2%；
- activation recall 至少 25%；
- 31 棵目标树的 proposal coverage recall 至少 50%；
- MLP 至少 2/3 个随机 seed 通过；
- 锁定 seed 42 必须通过。

验证保留率只搜索：

`0.0005, 0.001, 0.002, 0.005, 0.01, 0.02`

每个森林最多选择 250 个 proposal，并使用 0.3 m NMS。Q5b1 通过后仍不能声称性能提高，必须进入 Q5b2，在固定 validation forests 上真正补种、重跑 HDBSCAN 并检查 F1、Completeness、Commission 和丢树数。

## 6. 服务器执行流程

### 6.1 更新代码与测试

```bash
cd ~/projects/zrx/code/TreeLearn
git switch vertical-instance-quality
git pull --ff-only origin vertical-instance-quality

conda activate TreeLearn
mkdir -p logs/coverage_preserving_quality

python -m unittest \
  tests.test_seed_completion_proposals \
  tests.test_seed_completion_learning \
  tests.test_seed_completion_activation -v
```

### 6.2 先运行 pilot

Pilot 只运行 A1N，并导入 G4N 的 Q5b0 artifact：

```bash
set -o pipefail

python -u tools/data_gen/gen_seed_completion_learning_data.py \
  --config configs/experiments/coverage_preserving_quality/q5b1_generate_seed_completion_learning.yaml \
  --pilot \
  2>&1 | tee logs/coverage_preserving_quality/q5b1_data_pilot.log

cat data/seed_completion_learning/pilot_summary.md
```

只有 pilot Gate 全部为 True 才运行全量。

### 6.3 全量生成 13/5 数据

该步骤会长时间运行，可以用 nohup：

```bash
nohup python -u tools/data_gen/gen_seed_completion_learning_data.py \
  --config configs/experiments/coverage_preserving_quality/q5b1_generate_seed_completion_learning.yaml \
  > logs/coverage_preserving_quality/q5b1_data_full.log \
  2>&1 < /dev/null &

echo $! | tee logs/coverage_preserving_quality/q5b1_data_full.pid
tail -f logs/coverage_preserving_quality/q5b1_data_full.log
```

检查进程和结果：

```bash
pid=$(cat logs/coverage_preserving_quality/q5b1_data_full.pid)
ps -p "$pid" -o pid,%cpu,%mem,rss,etime,stat,cmd

cat data/seed_completion_learning/generation_summary.md
wc -l data/seed_completion_learning/manifest.csv
```

若日志显示 Gate 失败，停止，不训练激活头。不要通过修改 Gate 绕过失败。

### 6.4 训练并比较三个模型

```bash
nohup env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  python -u tools/training/train_seed_completion_activation.py \
  --config configs/experiments/coverage_preserving_quality/q5b1_train_seed_completion_activation.yaml \
  > logs/coverage_preserving_quality/q5b1_seed_completion_activation_run.log \
  2>&1 < /dev/null &

echo $! | tee logs/coverage_preserving_quality/q5b1_seed_completion_activation.pid
tail -f logs/coverage_preserving_quality/q5b1_seed_completion_activation_run.log
```

最终查看：

```bash
cat logs/coverage_preserving_quality/q5b1_seed_completion_activation/summary.md
cat logs/coverage_preserving_quality/q5b1_seed_completion_activation/per_run_metrics.csv
```

## 7. 继续与停止条件

- 若 Q5b1 Gate 通过：锁定推荐模型、seed、NMS 和 keep ratio，下一步实现 Q5b2 validation pipeline 集成。
- 若只有固定规则或 Logistic 通过：可以继续 Q5b2，但论文中不能声称 MLP 学习模块有效。
- 若三个模型都失败：关闭 Seed-Completion 路线，不使用 Wytham 调参。
- 在 Q5b2 validation Gate 通过之前，禁止运行 Wytham。
