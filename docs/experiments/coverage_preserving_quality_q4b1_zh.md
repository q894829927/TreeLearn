# Q4b1：Candidate + K + Safety 三头实例拆分控制实验

## 1. 出发点

Q4b0 已证明 Raw-XY known-K Oracle 拆分可以将固定 validation forests 的检测 F1 从
88.509% 提高到 89.308%（+0.798 pp），且不丢失原本检出的树。但移除 GT 验收后：

- known-K accept-all：F1 -0.450 pp，丢失 6 棵基线树；
- fixed-K2 accept-all：F1 -0.426 pp，丢失 6 棵基线树；
- Commission 均恶化约 1.7 pp。

因此 Q4b1 必须同时验证三个可部署预测头：

1. `candidate_head`：当前预测实例是否为 Q3 欠分割父实例；
2. `child_count_head`：拆成多少个子实例，固定类别 K=2、3、4；
3. `safety_head`：当前 Raw-XY K-means proposal 是否应被接受。

本阶段是参数较小的 MLP 控制实验，不实现注意力、不修改 TreeLearn 主干、不修改
HDBSCAN，也不读取 Wytham。

## 2. 数据与标签

沿用已经通过数据泄漏审计的固定划分：13 个 train forests 与 5 个 validation forests。
特征、标签和 proposal 必须在同一次 TreeLearn 推理中生成，不能用运行内
`instance_id` 拼接另一轮 artifact。

模型输入不含 GT：

- 35 维实例全局几何与置信度特征；
- 8 层垂直 token 的 masked mean 与 masked max；
- safety head 额外接收 16 维 Raw-XY proposal 几何特征。

GT 只产生监督标签和验证指标：

- Candidate：Q3 首个失败类别为 `undersegmentation` 的父实例；
- K：保留主分量并为每棵漏检树增加一个子实例，最大 K=4；
- Safety：子分区在“恢复树数、counted FP、accepted IoU”的字典序上优于父分区。

每片森林同时保存压缩的基线列联表及 proposal 子列联表。训练脚本将模型选择的多个
拆分组合后重新执行 Hungarian matching，精确计算 TP/FP/FN，而不是用分类 AP 代替最终
检测效果。

## 3. 科学 Gate

单个 seed 必须同时满足：

- validation F1 至少提高 0.5 pp；
- Completeness 最多下降 0.2 pp；
- Commission 最多增加 0.5 pp；
- 至少 4/5 validation forests 的 F1 不下降。

最终 Gate：

- seed 42 必须通过；
- 42、43、44 至少 2/3 通过；
- 三 seed F1 增益标准差不超过 0.30 pp。

PASS 才进入 Q4b2 pipeline 集成。FAIL 则关闭实例拆分路线，不用 Wytham 调阈值。

## 4. 本地/服务器测试

```bash
python -m unittest \
  tests.test_instance_split_diagnostics \
  tests.test_instance_split_oracle \
  tests.test_instance_split_deployability \
  tests.test_instance_split_learning \
  tests.test_instance_split_heads -v
```

## 5. Pilot 数据生成

Pilot 只运行 A1N 与 G4N，用于检查环境、显存、artifact 和 Q3 对齐：

```bash
conda activate TreeLearn
mkdir -p logs/coverage_preserving_quality
set -o pipefail

python -u tools/data_gen/gen_instance_split_learning_data.py \
  --config configs/experiments/coverage_preserving_quality/q4b1_generate_split_learning.yaml \
  --pilot \
  2>&1 | tee logs/coverage_preserving_quality/q4b1_data_pilot.log

cat data/instance_split_learning/pilot_summary.md
```

只有 Pilot Gate 全部为 True 才执行 full。Pilot artifact 会被 full 自动复用。

## 6. Full 数据生成

```bash
nohup python -u tools/data_gen/gen_instance_split_learning_data.py \
  --config configs/experiments/coverage_preserving_quality/q4b1_generate_split_learning.yaml \
  > logs/coverage_preserving_quality/q4b1_data_full.log \
  2>&1 < /dev/null &

echo $! | tee logs/coverage_preserving_quality/q4b1_data_full.pid
tail -f logs/coverage_preserving_quality/q4b1_data_full.log
```

预计约 6～12 小时，取决于 CPU 和 forest 大小。脚本支持断点续跑；不要删除已验证的
artifact，不要随意使用 `--force`。

查看当前 forest 的底层进度：

```bash
tail -f data/instance_split_learning/logs/G4N.log
```

完成后：

```bash
cat data/instance_split_learning/generation_summary.md
wc -l data/instance_split_learning/manifest.csv
```

完整数据 Gate 必须复现：validation candidate parents=69、undersegmented GT=73、
known-K safe parents=25，并且 K>4 overflow=0。

## 7. 三头 MLP 训练

```bash
nohup env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  python -u tools/training/train_instance_split_heads.py \
  --config configs/experiments/coverage_preserving_quality/q4b1_train_split_heads.yaml \
  > logs/coverage_preserving_quality/q4b1_split_heads_run.log \
  2>&1 < /dev/null &

echo $! | tee logs/coverage_preserving_quality/q4b1_split_heads.pid
tail -f logs/coverage_preserving_quality/q4b1_split_heads_run.log
```

判断是否结束：

```bash
pid=$(cat logs/coverage_preserving_quality/q4b1_split_heads.pid)
ps -p "$pid" -o pid,%cpu,%mem,rss,etime,stat,cmd
```

若 `ps` 只有表头或没有输出，进程已结束。查看结果：

```bash
cat logs/coverage_preserving_quality/q4b1_split_heads/summary.md
cat logs/coverage_preserving_quality/q4b1_split_heads/per_seed_metrics.csv
```

锁定 checkpoint 仅允许使用：

```text
logs/coverage_preserving_quality/q4b1_split_heads/checkpoints/split_heads_seed42.pth
```

不得根据 Wytham 结果重选 checkpoint、Candidate 阈值、Safety 阈值或 K 类别。

## 8. 固定验证集结果

Q4b1 三个 seed 均未通过：Candidate AP 为 0.087–0.112，Safety AP 为
0.196–0.320，K accuracy 为 0.899–0.913。最终 F1 仅提高 0.061–0.097 pp，
远低于预注册的 0.5 pp；虽然 Completeness 略有上升，但 Commission 同时增加
0.178–0.252 pp。实例拆分路线据此关闭，不再修改三头网络或阈值。

下一阶段转向 Q3 中尚未处理的 31 棵 `base_seed_support_failure`，先运行 Q5a
Coverage-Seed Completion Oracle；仍不读取 Wytham。
