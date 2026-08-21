# 垂直关系注意力微提议单木分割（Proposal-Relation Attention）

## 1. 研究目标

本路线从 `sparse-attention-unet` 分支冻结的 B1 Partial Fine-tune
checkpoint 出发：

```text
work_dirs/train_t1_b1_partial_seed42/best_unet.pth
```

不再改动点级主干，也不使用预测 offset、base seed 或 HDBSCAN 形成实例。
实例形成流程改为：

```text
B1 semantic tree probability
  -> 0.6 m XY cell + 0.5 m Z bin 垂直微提议
  -> 最多 16 邻居的稀疏图
  -> 两层垂直关系注意力
  -> reciprocal top-2 安全 union-find
  -> 单木实例
```

选择与训练只使用固定 13 个训练森林和 5 个验证森林。Wytham 只允许在
P0、P1、P2、L1W Gate 全部通过、模型/seed/阈值均锁定后运行一次。

## 2. 与历史失败路线的区别

此前后处理质量过滤、碎片合并、拆分、seed completion 和 Sparse U-Net
注意力的共同问题，是在 TreeLearn 已形成的错误实例或已丢失的 seed 上修补。
本路线将创新位置前移到“实例形成”：保留 B1 点级语义能力，但完全替换
offset-seed/HDBSCAN。因此它能同时表达纵向连续性、冠层尺度变化和相邻树间
hard negative，而不是只对最终实例打分。

失败路线的结论仍保留，不作为本路线的可调超参数来源。

## 3. 固定方法

微提议对 `tree_probability >= 0.5` 的点执行唯一归属：

- XY cell 0.6 m；
- Z bin 0.5 m；
- 同 XY cell 中超过 1.0 m 的垂直空断层切分；
- 不删除小节点；
- 节点坐标使用森林内相对位置，避免绝对坐标域偏移。

图边满足水平中心距离/包围盒、垂直 gap 和最大邻居约束。推理 graph
artifact 仅保存无标签数组；任何含 `gt/label/target/truth` 的字段都会被拒绝。

关系注意力中 edge embedding 同时进入 attention logit 和 value。训练损失为
source-balanced focal edge loss 与 GT-tree supervised contrastive loss。
解码只接受达到阈值、互为 top-2 且不违反组件直径/高度上限的边。

## 4. 文件

- 核心图与模型：`tree_learn/util/proposal_relation.py`
- 学习 artifact：`tree_learn/util/proposal_relation_learning.py`
- 锁定推理：`tree_learn/util/proposal_relation_inference.py`
- P0：`tools/diagnostics/diagnose_proposal_relation_oracle.py`
- P1：`tools/data_gen/gen_proposal_relation_data.py`
- P2：`tools/training/train_proposal_relation.py`
- P3 pipeline：`tools/pipeline/proposal_relation_pipeline.py`
- P3 汇总：`tools/diagnostics/summarize_proposal_relation_final.py`

## 5. 实验顺序

### P0：固定验证集 Oracle

```bash
mkdir -p logs/proposal_relation_attention
set -o pipefail

python -m unittest \
  tests.test_proposal_relation \
  tests.test_proposal_relation_attention -v

nohup python -u tools/diagnostics/diagnose_proposal_relation_oracle.py \
  --config configs/experiments/proposal_relation_attention/p0_micro_proposal_oracle.yaml \
  > logs/proposal_relation_attention/p0_run.log 2>&1 < /dev/null &

echo $! | tee logs/proposal_relation_attention/p0.pid
tail -f logs/proposal_relation_attention/p0_run.log
```

查看：

```bash
cat logs/proposal_relation_attention/p0_micro_proposal_oracle/summary.md
```

P0 Gate 固定为 F1 上限 +2.0 pp、Completeness +1.0 pp、恢复至少 20 棵、
Commission 不恶化、至少 4/5 森林非负、漏检图覆盖至少 90%、节点膨胀不超过
10 倍、平均出度不超过 16。失败立即停止，不运行 P1/P2。

### P1：固定森林图数据

仅 P0 PASS 后运行：

```bash
nohup python -u tools/data_gen/gen_proposal_relation_data.py \
  --config configs/experiments/proposal_relation_attention/p1_generate_graph_data.yaml \
  > logs/proposal_relation_attention/p1_run.log 2>&1 < /dev/null &

echo $! | tee logs/proposal_relation_attention/p1.pid
tail -f logs/proposal_relation_attention/p1_run.log
```

查看：

```bash
cat data/proposal_relation/generation_summary.md
wc -l data/proposal_relation/manifest.csv
```

### P2：参数匹配模型竞赛

仅 P1 PASS 后运行：

```bash
nohup env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  python -u tools/training/train_proposal_relation.py \
  --config configs/experiments/proposal_relation_attention/p2_train_models.yaml \
  > logs/proposal_relation_attention/p2_run.log 2>&1 < /dev/null &

echo $! | tee logs/proposal_relation_attention/p2.pid
tail -f logs/proposal_relation_attention/p2_run.log
```

比较固定几何、参数匹配 Edge-MLP、无垂直特征 Relation Attention 和
Vertical Relation Attention，训练 seeds 42/43/44。只有 M1 通过全部 Gate，
`summary.json` 才写入锁定 checkpoint、seed 与 threshold。

### P3：L1W 安全检查与 Wytham 锁定评价

先运行 L1W：

```bash
python -u tools/pipeline/proposal_relation_pipeline.py \
  --config configs/experiments/proposal_relation_attention/pipeline_l1w_locked.yaml \
  2>&1 | tee logs/proposal_relation_attention/pipeline_l1w_locked.log

python -u tools/evaluation/evaluate.py \
  --config configs/experiments/proposal_relation_attention/evaluate_l1w_locked.yaml \
  2>&1 | tee logs/proposal_relation_attention/evaluate_l1w_locked.log
```

确认 L1W Gate 后，先生成锁定的 B1-HDBSCAN 控制，再运行唯一 M1：

```bash
python -u tools/pipeline/pipeline.py \
  --config configs/experiments/proposal_relation_attention/pipeline_wytham_b1_hdbscan_locked.yaml \
  2>&1 | tee logs/proposal_relation_attention/pipeline_wytham_b1_hdbscan_locked.log

python -u tools/evaluation/evaluate.py \
  --config configs/experiments/proposal_relation_attention/evaluate_wytham_b1_hdbscan_locked.yaml \
  2>&1 | tee logs/proposal_relation_attention/evaluate_wytham_b1_hdbscan_locked.log

python -u tools/pipeline/proposal_relation_pipeline.py \
  --config configs/experiments/proposal_relation_attention/pipeline_wytham_locked.yaml \
  2>&1 | tee logs/proposal_relation_attention/pipeline_wytham_locked.log

python -u tools/evaluation/evaluate.py \
  --config configs/experiments/proposal_relation_attention/evaluate_wytham_locked.yaml \
  2>&1 | tee logs/proposal_relation_attention/evaluate_wytham_locked.log

python -u tools/diagnostics/summarize_proposal_relation_final.py \
  --config configs/experiments/proposal_relation_attention/p3_final_summary.yaml \
  2>&1 | tee logs/proposal_relation_attention/p3_summary_run.log
```

读取 Wytham 后禁止修改模型、阈值、图约束或聚合规则。

## 6. 论文记录要求

无论 Gate 成败均保留以下控制组：

- B1-HDBSCAN；
- fixed geometry；
- parameter-matched Edge-MLP；
- Relation Attention without vertical features；
- Vertical Relation Attention。

若 M1 不能击败 C1，只能报告“关系建模未证明有效”；若 P0 失败，则结论为
B1 点级语义/固定微提议缺少足够实例恢复上限。不得选择“最不差”的结果进入
Wytham。
