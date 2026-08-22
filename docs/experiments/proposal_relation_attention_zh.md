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

python -u tools/diagnostics/summarize_proposal_relation_final.py \
  --config configs/experiments/proposal_relation_attention/p3_final_summary.yaml \
  --l1w-only
```

只有 `--l1w-only` 返回 PASS，才生成锁定的 B1-HDBSCAN 控制并运行唯一 M1：

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

## 7. P0 失败复盘与 P0b 止损诊断

P0 原子微提议图恢复了 116 棵 B1 漏检树，漏检图覆盖为 98.193%，说明
B1 semantic 中仍保留了明显的漏检恢复信号；但 731,093 个节点相当于 B1
实例数的 369.986 倍，使 Oracle 自身产生大量碎片，F1 降低 9.221 pp，
Commission 上升到 30.454%。因此 P0 失败的主因是提议粒度，而不是没有
点级召回信号。禁止直接运行原 P1/P2。

P0b 是独立的新假设：在完全无标签的条件下先把原子节点压成
superproposals，再建立稀疏图并重新测 Oracle。它比较：

- `xy_block`：固定 3.6 m 水平块，作为强压缩控制；
- `vertical_profile`：严格的高度剖面、重叠和语义连续性连通；
- `adaptive_component`：按结构相似度贪心聚合，并限制组件直径和高度。

三种方法均直接读取现有 P0 artifacts，不重新跑 B1，也不读取 Wytham。Gate
仍沿用 P0：F1 +2.0 pp、Completeness +1.0 pp、至少恢复 20 棵、Commission
不恶化、4/5 森林非负、漏检覆盖 90%、节点膨胀不超过 10 倍、平均出度不超过
16。仅当至少一种方法完整通过，才为推荐方法重新设计 P1/P2 数据路径。

运行：

```bash
python -m unittest tests.test_proposal_relation_coarsening -v

nohup python -u tools/diagnostics/diagnose_hierarchical_proposal_oracle.py \
  --config configs/experiments/proposal_relation_attention/p0b_hierarchical_proposal_oracle.yaml \
  > logs/proposal_relation_attention/p0b_run.log 2>&1 < /dev/null &

echo $! | tee logs/proposal_relation_attention/p0b.pid
tail -f logs/proposal_relation_attention/p0b_run.log
```

查看：

```bash
cat logs/proposal_relation_attention/p0b_hierarchical_proposal_oracle/summary.md
```

P0b 失败时关闭整条 proposal-relation 路线，不从三个失败方法里选择“最不差”
方案，也不得通过 Wytham 反向调聚合尺度。

## 8. P0b 最终结果（2026-08-22）

P0b 使用固定三方法和原 P0 artifacts 完成止损诊断：

| Method | 已评价森林 | 节点/膨胀 | 结果 |
|---|---:|---:|---|
| xy_block | 2/5 | 2,594 / 2.80x | F1 相对 B1 -38.609 pp；强压缩导致不同树不可逆混合 |
| vertical_profile | 2/5 | 173,819 / 187.71x | F1 相对 B1 -44.872 pp；结构连续条件无法消除原子碎片 |
| adaptive_component | 0/5 | G4N 单森林仍有 27,537 节点 | 已超过五森林合计允许的 19,760 节点上限，建图前淘汰 |

Primary Gate 为 False，推荐方法为 None。由此得到的最终结论是：该表示不存在
同时满足“节点纯度、压缩率和完整实例连通性”的可用工作点。强压缩在学习前
已经混合相邻树，关系注意力不能再拆开；保守压缩则保留数万至数十万碎片，
既不满足部署复杂度，也无法控制 Commission。

因此从本日期起正式关闭以下内容：

- 不运行原 P1 数据生成；
- 不运行 P2 Edge-MLP/Relation-Attention 训练；
- 不为 P0b 调整 cell、profile 或 component 阈值；
- 不读取 Wytham 来反向选择预聚合方法。

该失败作为论文中的实例形成消融与方法边界保留。后续工作转入论文收口：
冻结 B0 Official、B1 Partial Fine-tune 和 M4 Window Attention，若开展 Wytham
实验，必须三者完整报告且明确标为锁定后的探索性跨域评价，不得据此继续调参。
