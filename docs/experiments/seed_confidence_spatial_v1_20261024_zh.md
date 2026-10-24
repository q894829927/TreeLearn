# Seed 置信度筛选后续实验：空间分层 Top-78%（V1，待服务器评价）

日期：2026-10-24。独立分支：`codex/seed-confidence-spatial-v1`。
本文件为**实验设计与实现说明**；不包含新的源域或 Wytham F1，不能宣称当前改进有效。

## 已验证起点

基于 [MLP Seed 筛选历史最终记录](final_mlp_confidence_results_zh.md)：
冻结官方 TreeLearn 主干与 heads；轻量 MLP 输出 base-residual XY 和异方差置信度；
**不使用预测残差融合**，仅用置信度从原有 base seed 中选择 Top-78%，
仍采用原版 HDBSCAN 和剩余点归属。

- 已锁定模型：`work_dirs/base_residual_mlp_frozen_s43/best_base_residual_xy.pth`（仅是历史服务器路径，本分支未重新训练或验证该文件）。
- 历史 Wytham 独立口径：原版 F1 72.1%、MLP confidence s43 F1 73.9%，Commission 18.7%→13.5%，Completeness 64.8%→64.5%，Coverage 57.7%→56.8%。s42 F1 74.3%，但不是历史锁定模型。
- **这组历史数字与 2026-10 PPT 所用的 70.324% 基线协议不同，绝不可直接相减或混排**；新实验必须在同一份数据、预处理、权重和评价脚本中成对对比。

## 本轮唯一改变：Seed 选点时的排序范围

原模式 `top_ratio`：全部 base seeds 全局按 MLP 置信度降序排列，只保留前 78%。

新模式 `spatial_top_ratio`：按**原版 base XY 投票位置**（`coords[:, :2]+offset[:, :2]`）
划分固定尺寸的 XY 网格格子；在每个格子内部按**同一份 MLP 置信度**排序。
使用 Hamilton 最大余数法，按格子的原候选 Seed 数量比例分配名额；
所有格子的保留总数严格等于 `ceil(原 base seed 总数 × 0.78)`。
同分按原始点索引固定排序，不使用 GT、预测实例 ID、森林名称或 Wytham 特征。
对于没有足够名额的极小格子，**没有每格至少一粒 seed 的保证**，也不保证满足
HDBSCAN `tau_min=50`。这属于待检验假设，而非已解决的覆盖保护问题。

修改限定在 `tree_learn/util/axis.py`、`tree_learn/util/pipeline.py`、
`tools/pipeline/pipeline.py` 和 grouping 新的可选配置。旧模式计算分支保持不变；
默认 `use_seed_confidence_filter: False`，因此原版默认行为不变。
新增单独测试 `tests/test_seed_confidence_spatial.py`。

## 可复现实验组

所有实验使用**同一个已经锁定的 s43 模型 checkpoint**，不得重新训练或借 Wytham 调整
置信度、保留比例、后处理、tree confidence、Verticality、Offset Z 或 HDBSCAN。

| 组别 | mode | keep_ratio | cell_size | 用途 |
|---|---|---:|---:|---|
| C0 | threshold / 关闭筛选 | 1.00 | — | 原版全 seed |
| C1 | random_ratio | 0.78 | — | 等量随机删除 |
| C2（历史锁定） | top_ratio | 0.78 | — | 主对照 |
| C3（本轮） | spatial_top_ratio | 0.78 | 2.0m | 预设新方法 |

**第一轮只测 C2 vs C3，固定 cell_size=2.0m。** 如果 C3 失败，则先检查失效机制，不允许看到 Wytham 的结果后再扫 cell size。
1.0m/4.0m 仅作为后续独立的源域敏感性分析，使用同一历史模型。
实际评估数据优先使用历史的 5 个开发森林 G4N/G4W/L1N/O1N/O1W；Wytham 只作已经被多轮使用过的探索性跨域确认，不当作新的盲测。

## 如何运行

```bash
git fetch origin
git switch codex/seed-confidence-spatial-v1
git pull --ff-only origin codex/seed-confidence-spatial-v1
python -m unittest discover -s tests -p 'test_seed_confidence_spatial.py' -v
```

从**之前已经跑通、可加载锁定 s43 MLP checkpoint 的 pipeline 配置**复制两份配置
（比如 `configs/pipeline/<existing-working-config>.yaml`；不要假定仓库默认
`pipeline.yaml` 自动加载这一额外 MLP checkpoint）。
两份配置保持 forest、pretrain、model、dataset、阈值、tile 规则完全一致，仅更改：

```yaml
grouping:
  use_axis_fusion: False
  use_upper_seeds: False
  use_seed_confidence_filter: True
  seed_confidence_keep_ratio: 0.78
  seed_confidence_filter_mode: 'top_ratio'  # C2
```

C3 则把最后一项改为 `spatial_top_ratio`，增加
`seed_confidence_spatial_cell_size: 2.0`。
**保留之前实际生效的其余 grouping、model、输入与 checkpoint 配置，不得用上面的小片段整体覆盖整个 grouping。**
不同组必须使用不同结果目录，避免覆盖预测。运行形式：

```bash
python tools/pipeline/pipeline.py --config <完整的C2配置路径>
python tools/pipeline/pipeline.py --config <完整的C3配置路径>
```

执行前核对锁定 checkpoint 能正确加载（至少 `axis_confidence` 非空且为有限数）；
C2/C3 中输入原候选数应完全相同，保留 Seed 数也必须完全相同。
若不一致应停止评价，首先排查配置或数据。

## 必须保存的实验结果

1. 五森林逐林 TP/FP/FN、检测 F1、Completeness、Commission、Coverage、
   点级 Precision/Recall、预测实例数；
2. 原 seed 数、保留 seed 数及按 base-vote XY 格子的保留率分布；
3. 用**评价阶段才读取**的 GT 统计逐棵树保留 Seed 数和变化，尤其检查
   原本检出而变漏检、原本漏检而恢复的树；
4. 按真实树的 Z span（不是地面归一化树高）分层汇总，同时另存
   offset/verticality 和邻树密度诊断；
5. 原版 TP 身份损失/新恢复、相同 seed 数但不同位置的实例数变化；
6. 完整模型权重 SHA、源数据与 GT 哈希、配置 SHA、代码 commit、运行耗时和最大显存。

## 提前固定的决策规则

- 新组 C3 必须在**源域五森林**上宏 F1 不低于 C2，至少 4/5 森林不下降，
  而且不能靠明显牺牲 Completeness 或 Coverage 换取 FP 降低；
- 至少分析一次全部差异 GT 的种子保留和匹配身份，而不仅仅比较平均 F1；
- 不通过就保留 C2 为最终方法，记录 C3 失败，**不根据 Wytham 调阈值**；
- 源域满足后再对 Wytham 一次固定参数外推，仍须第三个此前未用的森林做真正独立泛化评价。

**当前状态：新筛选代码与 CPU 单元测试文件已提交；未跑服务器测试、整林推理或新指标。**
