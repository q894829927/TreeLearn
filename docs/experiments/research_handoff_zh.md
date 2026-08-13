# TreeLearn 复杂森林单木分割：跨对话研究交接

> 最后更新：2026-08-13
> 当前分支：`vertical-instance-quality`
> 当前提交：`45d269f Add Q6a fragment graph oracle`
> 用途：新对话、换设备或中断恢复时，先阅读本文件，不要重新尝试已关闭路线。

## 1. 新对话启动提示

将下面内容发给新的 Codex 对话：

```text
请先完整阅读 docs/experiments/research_handoff_zh.md，并以它作为当前研究状态的
唯一交接依据。不要把计划当成已完成结果，不要重复已经被 Gate 关闭的实验。

Q6a fragment graph Oracle 已完成并按预注册 Gate 关闭；先核对交接文档中的结果，
然后进入冻结 TreeLearn 主干的 HSCA 高度分层上下文注意力实验。若需要
新增代码，请先检查当前分支、已有接口和未提交修改，再按文档中的 A0-A4 顺序
实现、测试并给出服务器命令。
```

## 2. 研究目标与现实约束

### 2.1 毕业论文目标

研究目标是改进 TreeLearn 在树干遮挡、冠层重叠、复层结构和密度变化明显的
复杂森林中的单木点云实例分割，并保留真实、可解释的注意力机制作为开题报告中
的技术内容。

推荐论文定位：

> 面向复杂森林场景的高度分层上下文注意力单木点云实例分割方法

“少样本适配”只作为时间允许时的后续扩展；在没有独立实验和跨区域划分前，不写入
论文题目或主要贡献。

不再追求从零复现一个大型 Transformer，也不把普通门控或 MLP 包装成注意力。

### 2.2 算力与时间

- 当前主要训练设备：RTX 4090 24 GB；
- 临近毕业论文写作，优先选择低风险、可复现、训练成本可控的方案；
- ForestFormer3D 默认使用半径 16 m、最多 640k 点、300 queries、6 层 decoder、
  batch size 2，并在 A100 上训练，因此不作为当前主工程框架；
- 继续使用 TreeLearn，冻结其 sparse U-Net 主干，通过轻量注意力适配器训练。

### 2.3 数据角色

- `L1W`：性能接近饱和，作为源域回归检查和历史开发基准；
- `Wytham`：已经被多次查看并参与方法判断，只能描述为复杂林开发/回顾性基准，
  不能再声称完全未见测试集；
- 固定训练森林：A1N/A1W、G1N/G1W、G2N/G2W、G3N/G3W、L2N/L2W、
  LG1/LG2/LG3；
- 固定验证森林：G4N、G4W、L1N、O1N、O1W；
- 正式论文最好再增加 LAUTx、FOR-instanceV2 中未参与设计的区域，或一个新的完整
  标注复杂森林作为锁定外部测试集。

## 3. 当前代码与关键资产

### 3.1 关键路径

```text
官方/主要初始化权重：
data/model_weights/model_weights_with_small_20241213.pth

当前锁定的 seed-confidence checkpoint：
work_dirs/base_residual_mlp_frozen_s43/best_base_residual_xy.pth

Wytham 输入森林：
data/pipeline/wytham/forest/wytham_vox0.1.laz

Wytham GT：
data/benchmark/wytham_vox0.1.laz

Q6a 配置：
configs/experiments/coverage_preserving_quality/q6a_fragment_graph_oracle.yaml

Q6a 主日志：
logs/coverage_preserving_quality/q6a_fragment_graph_oracle_run.log
```

### 3.2 已有模型接口

`tree_learn/model/tree_learn.py` 当前已经包含：

- sparse U-Net 主干；
- semantic head 和 base-offset head；
- upper-offset 分支；
- MLP/Local Point Transformer axis 分支；
- 冻结模块及冻结 BatchNorm 的机制；
- `backbone_feats`、点坐标、verticality、batch id 等 HSCA 所需输入。

已有 `tree_learn/util/seed_attention.py` 和
`tree_learn/model/point_transformer.py` 可参考 Q/K/V、相对几何、mask、残差和测试
写法，但旧 seed attention/axis attention 的结果不能作为 HSCA 成功证据。

## 4. 已确认的基准结果

### 4.1 TreeLearn 原始基线

| 数据集 | Completeness | Commission | Detection F1 | 点级 Precision | 点级 Recall | Coverage |
|---|---:|---:|---:|---:|---:|---:|
| L1W base/r100 | 100.0% | 3.1% | 98.4% | 98.9% | 99.1% | 98.0% |
| Wytham base/r100 | 64.8% | 18.7% | 72.1% | 62.5% | 80.5% | 57.7% |

注意：日志中的 `Precision/Recall/Coverage` 属于点级分割评价；检测 Precision 应由
`1 - Commission` 得到。论文中不得混用二者。

### 4.2 当前最有效的已完成改进

冻结 TreeLearn 主干和原始 heads，以 MLP 预测 base-residual 置信度，只按排名保留
前 78% base seeds，不修改 vote 坐标：

| 方法 | Completeness | Commission | Detection F1 | 点级 Precision | 点级 Recall | Coverage |
|---|---:|---:|---:|---:|---:|---:|
| L1W MLP-confidence r078，seed 43 | 100.0% | 0.6% | 99.7% | 98.9% | 99.1% | 98.0% |
| Wytham MLP-confidence r078，seed 43 locked | 64.5% | 13.5% | 73.9% | 60.3% | 80.9% | 56.8% |
| Wytham MLP-confidence r078，seed 42 | 64.5% | 12.5% | 74.3% | 60.0% | 81.1% | 56.7% |

锁定 seed 43 相对 Wytham baseline：Detection F1 `+1.8 pp`、Commission
`-5.2 pp`、Completeness `-0.3 pp`，但点级 Precision `-2.2 pp`、Coverage
`-0.9 pp`。因此只能声称减少假阳性并提升检测 F1，不能声称分割质量全面提高。

三个训练随机种子的 L1W F1 为 99.7%、99.7%、100.0%，且优于随机等量保留
78% seeds 的约 98.8% 平均 F1，证明 learned ranking 不只是 seed 数量减少。

## 5. 已完成实验与科学结论

| 路线 | 主要结果 | 当前结论 |
|---|---|---|
| Upper/stem Axis + Point Transformer | PT 相对 MLP 的 Axis mean error 仅改善约 1.4%；融合权重在 L1W 无收益 | 关闭，不再增加轴分支层数 |
| Base residual MLP confidence | L1W 稳定改善；Wytham Detection F1 提升 1.8 pp | 可作为已有成果/基线模块 |
| 绝对 confidence 阈值 | L1W 保留 78.2% seeds，Wytham仅保留48.2%，Wytham F1降到69.8% | 存在跨域校准漂移，禁止继续扫绝对阈值 |
| 实例质量 Vertical-MLP | L1W过滤后F1 99.68%；Wytham固定阈值导致F1下降 | 固定阈值跨域失败 |
| Top-ratio 实例质量过滤 | Wytham r085 F1 73.23%，Commission降低5.85 pp，但Completeness下降1.60 pp | 可作风险-覆盖工作点，不作全面提升结论 |
| 选择性实例分割 E9 | Wytham相对随机排序：Commission面积降低41.32%，F1面积增加10.29 pp；置换检验 `p=0.0001` | 质量排序有效，可作为论文辅助贡献 |
| Vertical vs Global E10/E10b | 垂直模型没有显著优于参数匹配 Global-MLP；Fusion也未带来稳定选择性曲线增益 | 不得声称垂直 token 优于全局特征 |
| 实例合并 E8 | Oracle 有空间，但 pair/groupwise 模型在安全约束下召回约1% | 学习式合并关闭 |
| Seed MLP/关系注意力 | Reliability AUC Gate失败；关系注意力未证明优势 | 旧 seed-attention 路线关闭 |
| Cellwise vote refinement | Oracle大幅降低vote误差，但碰撞增加；拓扑安全门控又损害F1 | vote refinement关闭 |
| 多尺度HDBSCAN | Proposal Oracle恢复不足；mcs100在L1W好但Wytham F1仅71.6% | 复杂度感知分组关闭 |
| Coverage-CVaR Q1-Q2 | Q1 Oracle通过；CVaR不优于参数匹配 safe+IoU MLP | CVaR/关系注意力不继续 |
| 实例拆分 Q4 | 几何 Oracle约+0.8 pp；Candidate+K+Safety 三头网络仅+0.077 pp | 拆分路线关闭 |
| Seed completion Q5 | Oracle约+0.5 pp；激活正样本极少，activation precision约1% | completion路线关闭 |
| Fragment graph Q6a | 图覆盖33/41个目标，只恢复9棵；F1 `+0.827 pp`，无TP丢失，Commission下降，但恢复数Gate失败 | 关闭fragmentation合并，不进入Q6b |

总体诊断：主要瓶颈是 TreeLearn 的实例形成机制和复杂林跨域表征，而不是缺少更多
后处理 MLP、阈值或 HDBSCAN 参数。Oracle 上限经常存在，但稀有关键事件难以从
现有统计特征中可靠识别。

## 6. Q6a 已完成：关闭 Fragmentation 合并路线

Q6a 使用固定五个 validation forests，未读取 Wytham。它判断仅根据预测实例的
几何与 base-vote 邻接关系，是否存在合并 fragmentation 碎片的上限。

原预注册 Gate：

- 精确复现 41 个 Q3 fragmentation 目标；
- 理想 union 至少恢复 20 棵、F1 至少 `+0.70 pp`；
- GT-free 图至少覆盖 25 棵；
- 图 Oracle 至少恢复 10 棵、F1 至少 `+0.50 pp`；
- 不丢失已有 TP，Commission 不增加；
- 至少 4/5 个验证森林 F1 不下降。

实际结果：

| Mode | TP | FP | FN | Completeness | Commission | F1 | Recovered | Lost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 1633 | 275 | 149 | 91.639% | 14.413% | 88.509% | - | - |
| fragment union ceiling | 1641 | 255 | 141 | 92.088% | 13.449% | 89.233% | 8 | 0 |
| fragment graph Oracle | 1642 | 252 | 140 | 92.144% | 13.305% | 89.336% | 9 | 0 |

图 Oracle 相对 baseline：F1 `+0.827 pp`、Completeness `+0.505 pp`、Commission
`-1.108 pp`，且没有丢失已有 TP。GT-free 图覆盖33/41个目标，但最终只恢复9棵；
恢复只发生在 L1N（3棵）和 O1W（6棵），G4N/G4W均为0。

结论：`union_ceiling_passed=False`、`graph_recovery_passed=False`，总 Gate 失败。
恢复数低于预注册的10棵，也低于保留为辅助后处理所要求的15棵。因此固定执行：

- 关闭 fragmentation merge；
- 不实现 Q6b 学习头；
- 不在 Wytham 上测试或调参；
- 结果只作为“Oracle 有有限空间、事件稀少且森林间不稳定”的瓶颈分析；
- 下一条主线直接进入 HSCA A0/A1/A2。

## 7. 下一条主线：HSCA 注意力适配器

### 7.1 模块定义

名称：`Height-Stratified Context Attention (HSCA)`，中文为“高度分层上下文
注意力”。插入位置是冻结 sparse U-Net 的逐点 `backbone_feats` 与预测结果之间，
它直接学习 semantic/base-offset 残差，不再预测无效的 upper axis。

```text
冻结 TreeLearn sparse U-Net 与原始 heads
        ↓
backbone feature [N, 32]
        + verticality
        + 由原始 offset-z 估计的归一化高度
        ↓
每个 tile 划分 8 个高度层
        ↓
高度层 mean/max pooling → 8 个 token
        ↓
单层 4-head self-attention（d_model=32）
        ↓
按高度层把上下文映射回逐点特征
        ↓
sigmoid gate + residual adapter
        ↓
原始 semantic logits + semantic residual
原始 base offsets + offset residual
```

固定默认值：

- `num_height_bins=8`；
- `hidden_dim=32`；
- `num_heads=4`；
- `num_layers=1`；
- `dropout=0.1`；
- 高度为 `relu(-base_offset_z)`，每个样本按 P95 归一化到 `[0,1]`；
- 空高度层使用 mask，不参与 attention softmax；
- residual heads 零初始化；残差缩放初始化为1，以保持初始输出等价且避免零梯度；
- 官方 checkpoint 加载后的初始输出必须与原始 TreeLearn逐元素一致；
- 第一阶段永久冻结 `input_conv/unet/output_layer/semantic_linear/offset_linear`，
  只训练 HSCA、gate 和两个 residual heads；
- attention 只处理每个样本8个token，4090显存开销应很小。

### 7.2 参数匹配控制组

必须实现相近参数量的 `Height-Stratified MLP Adapter`：使用相同高度层、相同
mean/max token 和相同 residual heads，但用两层 MLP 聚合，不进行 token 间 Q/K/V
注意力。

只有 HSCA 稳定优于该 MLP，才能把提升归因于注意力；不能只与原始 TreeLearn
比较。

### 7.3 实验顺序

| 编号 | 实验 | 目的 |
|---|---|---|
| A0 | 官方/当前 TreeLearn | 原始基线 |
| A1 | 冻结主干 + 参数匹配高度MLP adapter | 排除仅增加参数的收益 |
| A2 | 冻结主干 + HSCA | 验证真实注意力增益 |
| A3 | HSCA + SODA遮挡/密度退化增强 | 复杂林最终候选模型 |
| A4 | A3 + 已有实例质量排序 | 作为可选风险控制输出，不改变主分割结论 |

先只运行 seed 42 和一个固定 validation 流程：

1. 完成单元测试与 checkpoint 等价性；
2. A1、A2 使用相同数据、epoch、优化器和可训练参数量；
3. A2 通过快速 Gate 后才运行 A3；
4. 最终候选模型再运行 seed 42/43/44；
5. 所有 checkpoint 和超参数只根据固定 validation forests 决定；
6. 锁定后最后运行 Wytham，不再根据其结果修改配置。

快速 Gate：

- A2 相对 A1：validation macro Detection F1 至少 `+0.3 pp`，或 Coverage至少
  `+0.5 pp`；
- 至少3/5 validation forests同方向提升；
- L1W Detection F1下降不超过`0.5 pp`；
- 三个随机种子正式结果的多数方向一致。

若 A2未过 Gate，注意力仍可作为真实负消融写入论文，但不得继续扩大网络或从
Wytham挑参数。

### 7.4 SODA 数据增强

只有 A2 通过后实现 `Stem-Occlusion and Density Degradation Augmentation`：

- 随机删除离地约0-2 m范围内20%-80%的点，模拟树干遮挡；
- 随机删除局部竖直柱状区域；
- 进行高度/距离相关稀疏化，模拟远距离冠层与扫描缺口；
- 所有增强只用于训练，验证和测试保持原始点云；
- A3必须与不含HSCA、但使用相同增强的控制组比较，避免把增强收益归给注意力。

最终最低成功条件：Wytham Detection F1 相对72.08%提高至少`0.5 pp`，或
Coverage提高至少`1.0 pp`，且L1W Detection F1下降不超过`0.5 pp`。若只有
Commission改善，应明确报告 Completeness、点级 Precision 和 Coverage 的代价。

## 8. 论文可用贡献与禁止表述

### 8.1 可形成的贡献

1. 系统分析 TreeLearn 在复杂林中由树干遮挡、密度变化和实例形成引起的跨域退化；
2. 提出轻量 HSCA，通过高度层之间的 Q/K/V 注意力增强冻结 TreeLearn 表征；
3. 提出 SODA 模拟树干遮挡和密度退化，研究其与 HSCA 的协同；
4. 保留已有质量排序作为选择性风险控制，提供完整实例与高可信实例两个工作点。

### 8.2 必须诚实保留的限制

- Wytham不是完全未见测试集；
- Vertical-MLP没有证明优于Global-MLP；
- 旧Point Transformer/seed attention没有带来下游稳定提升；
- 多条Oracle路线有理论空间，但可部署学习头失败；
- 若HSCA不优于参数匹配MLP，论文不能声称注意力有效。

禁止只报告最好随机种子、混用指标、事后扫描Wytham参数、隐藏负结果，或把普通
sigmoid gate称为Transformer attention。

## 9. 新对话的行动清单

```text
[ ] 读取本文件和 git status，保护用户已有修改
[x] Q6a 已完成并关闭；不进入 Q6b，也不使用 Wytham 调参
[x] 已新建 `docs/experiments/height_context_attention_zh.md` 和独立配置目录
[x] 已实现 Height-Stratified MLP 与 HSCA 两个参数匹配 adapter
[x] 已添加空层mask、batch隔离、FP16、梯度和checkpoint等价性测试
[ ] 先跑 A0/A1/A2 seed42，不提前跑 Wytham
[ ] A2通过后实现并验证 SODA
[ ] 锁定候选后跑3个seed和Wytham
[ ] 汇总精确指标、显存、运行时间、参数量及失败分析
```

任何新路线都应先提出明确 Oracle 或快速 Gate；除 Q6a 外，不再延续 Q1-Q6 的
后处理探索链。
