# PTv3 × TreeLearn 论文核对与下一步计划（2026-10-24）

建议先验证“模型能否把同树点投到同一基部”，再决定新的训练预算。
当前最值得增加到诊断中的变量是PTv3的block学习率倍率；最需要细化的指标是Seed点的投票可分性。
本轮完成文献阅读、代码核对和CPU checkpoint元数据检查，没有启动训练、前向、反向或新的评价。
下列GPU测试及训练步数均为后续计划，未执行。

## 阅读版本与主要依据

- [PTv3：Point Transformer V3: Simpler, Faster, Stronger](https://arxiv.org/html/2312.10035v2)，CVPR 2024对应作者稿v2，重点核对第3—6节、附录A/B。
- [TreeLearn：A deep learning method for segmenting individual trees from ground-based LiDAR forest point clouds](https://arxiv.org/html/2309.08471v3)，Ecological Informatics 84 (2024), 102888的作者稿v3，核对方法、训练、评价、错误分析及附录。
- [PTv3作者2024年ScanNet配置](https://github.com/Pointcept/Pointcept/blob/8c3b8ff09c8879fad0e539c7e3f54c5336014bcc/configs/scannet/semseg-pt-v3m1-0-base.py)，固定commit `8c3b8ff`。不把当前main配置当作2024论文的逐项复现。
- 本项目冻结main审计、5000完整快照、[BN校准结果](../ptv3_bn_calibration_v1/README_zh.md)及用户当前进度文档。

PTv3的关键机制是几何序列化、多顺序patch交互和稀疏卷积xCPE。
论文采用block内LN/pre-norm，同时保留pooling BN；附录训练表列出block的0.1学习率倍率。
这支持检查优化和归一化的适配，不能直接推出“全部BN换LN”或“必须加入XYZ特征”。
[依据：PTv3第4节与附录A/B](https://arxiv.org/html/2312.10035v2)。

TreeLearn的实例能力依赖基部投票；错误投票会在邻树之间形成连接或产生虚假簇。
尤其需要修正此前对整体XY误差的解读：论文附录D的高分L1W模型XY误差也约2.25—2.30m。
当前2.84m与其点集、anchor、评价范围不同，不能直接比较，也不能拿聚类半径作为全体点误差门槛。
应分开检查树干Seed和树冠点。[依据：TreeLearn第2.2节、第3.4节、附录D](https://arxiv.org/html/2309.08471v3)。

## 论文、作者代码与本项目的边界

| 项目 | 文献/作者实现 | 当前冻结实验 | 处理原则 |
|---|---|---|---|
| PTv3 block LR | 论文为base的0.1；2024作者配置为0.006/0.0006 | 所有参数一个组，没有block倍率 | 作为独立优化配方假设验证，不能直接宣称是bug |
| PTv3 normalization | block LN，pooling BN | 与这一分工相容，另有原TreeLearn Head BN | 分模块定位，不整体换LN |
| PTv3空间信息 | 序列化和xCPE | 4通道常数特征，但grid、稀疏CPE和序列化均保留 | 常数输入不等于没有几何；测实际特征再判断 |
| PTv3 decoder depth | 作者稿表15为[1,1,1,1]，固定作者代码为[2,2,2,2] | [2,2,2,2] | 按来源记录差异，保留当前架构 |
| TreeLearn anchor | 树干约3m高的竖直点 | 完整增强crop的底部0.5m内点均值 | 继续以冻结main真实代码为准 |
| TreeLearn Offset Head | 论文写三层MLP | main为两层MLP、三维输出 | 不因论文而改Head或把loss改成二维 |
| TreeLearn聚类/传播 | 论文固定半径图聚类，最小100点，k=10 | HDBSCAN最小50点，k=5 | 保留当前冻结后处理 |
| TreeLearn训练 | 论文预训练150k、warmup6250且使用已有3D权重 | main预算250k、warmup12500；E1随机初始化 | 当前5000节点不能直接对标成熟官方权重 |

论文列出的Offset并非只有XY两个输出：其Z分量参与Seed条件，附录D也报告XYZ与XY损失。
聚类使用XY投影，不意味着应删除Z监督。
上述TreeLearn历史差异以[论文](https://arxiv.org/html/2309.08471v3)与本地
[冻结main审计](../ptv3_treelearn/source_audit_zh.md)交叉核对。
PTv3表格与作者发布配置的差异依据[固定2024代码](https://github.com/Pointcept/Pointcept/blob/8c3b8ff09c8879fad0e539c7e3f54c5336014bcc/configs/scannet/semseg-pt-v3m1-0-base.py)。

CPU读取原5000完整快照确认：459组参数张量只有1个optimizer group，当前lr=0.001206、
initial_lr=0.003、weight_decay=0.001；没有block专属倍率。
在同一base LR下，有无0.1倍率会使block LR相差10倍；这不是“当前LR恰好比论文高10倍”的结论。
不同任务的base LR、scheduler、batch和输入均不同，不能整体照搬ScanNet配方。
实际读取结果见 [optimizer_u5000.json](evidence/optimizer_u5000.json)。

## 目前能确定与不能确定的内容

| 已有观察 | 当前可支持的判断 | 尚不能支持的判断 |
|---|---|---|
| BN校准后F1从0到12.90%，2/24 TP；XYZ从334.54m到5.12m | 原推理统计失配是重要因素 | BN已经解决全部实例问题 |
| 语义树召回99.34%，Seed仅3527，区域聚类13个实例 | 需要继续追踪Seed与投票/归属链条 | 用语义召回证明实例有效 |
| 预测XY幅度很小，整体XY误差2.837m | 需要零/均值预测及分层误差对照 | 单凭平均误差认定特征塌缩 |
| block LR倍率与作者配方不同 | 有根据地安排单变量对照 | 未实测就把失败归因于学习率 |
| 仍在warmup内；没有E0-scratch同预算完整结果 | 现在无法评价Backbone优劣 | 自动训练到12500或250000就一定会成功 |

## 第一步：零optimizer更新的定点诊断

固定3个训练源区域、每区域2个既有完整crop，共6个，覆盖不同点数和树冠交叠程度。
固定原增强seed、voxel/point map、原5000权重及随机条件；按数据元信息选取，不按模型结果挑案例。
复用已完成的GT和点序测试，不重复冷读整个25000-crop池。

主要输出：

1. **监督与零预测对照。** 核查增强后anchor−xyz、单位、有效mask、voxel→point对应关系；
   同时计算零Offset和逐crop均值Offset的XY误差。
   报告预测/GT的逐轴均值、标准差、方向、误差分位及相对零预测的改善，不只看最大幅值。
2. **按角色分层。** 全部有效树点、GT定义的近基部竖直点、模型实际Seed分别统计XY/Z误差。
   GT分组只用于事后分析；不能偷偷变成模型的Seed。
   以main anchor的相对高度或Z Extent分层，不把它称为地面归一化树高。
3. **直接看投票。** 在 `q_xy = p_xy + offset_xy` 空间画同树投票、相邻树基部、
   类内散度和类间间隔；记录每棵树是否有Seed、Seed纯度、漏检/合并/碎裂。
4. **区分训练与推理。** 比较原running统计、既有静态校准统计、训练批统计诊断。
   DropPath和BN要分开控制；批统计分支不写回buffer，不是新正式评价。
5. **测特征与优化状态。** 记录CPE/attention/MLP/decoder残差的RMS、通道方差、层间幅度增长；
   按同树/异树比较特征。检查patch几何跨度和pooling中的实例混合，解释实际信息保留。
   使用索引和流式统计，不展开完整attention矩阵。

仅在需要梯度归因时，对最多2个代表crop分别重放semantic/offset的独立反向，
保持原Backbone AMP/Head FP32条件，无optimizer.step。
量化50×semantic与三维L2对各模块的梯度贡献，以及X/Y/Z输出梯度；
对 `||e_xyz||` 的梯度分量为 `e_axis/||e_xyz||`，Z主导是否压低XY学习必须实测。
不能仅凭三个轴量纲相同就假定其优化影响相同，也不在此步骤修改loss权重。

最多选2例做缓存预测数组的GT语义、GT XY、GT Z替换，共不超过6次聚类诊断。
未知标签保留未知；这类反事实只归因错误，不进入模型成绩，也不称严格性能上界。

建议预算：18次常规完整前向以内，另最多4次独立无更新反向和6次CPU聚类重放。
预估运行10—20分钟，开发和I/O另记；真实成本按第一例测量。
本步骤优先级最高，完成前不决定继续训练。

## 第二步：受限的小样本可学习性与LR对照

根据第一步证据选择下一项，每次只解决一个问题：

| 情况 | 下一项试验 | 单次上限和观察节点 |
|---|---|---|
| 监督/映射或mask不正确 | 先修独立适配器并做回归 | 不增加训练预算 |
| 训练批统计下投票正常，静态推理差 | 继续隔离BN/DropPath分布变化 | 先做无更新对照 |
| 训练批统计下XY也接近零/均值 | 固定2个完整crop的微型拟合 | 50/100/200步检查，最多200步 |
| 看到block残差/更新尺度异常 | 相同起点的block倍率1.0 vs 0.1单变量对照 | 两臂各最多200步，保持其余设置 |
| PTv3微型拟合仍不过 | 增加U-Net scratch的管线自检 | 独立最多200步，先验证显存可运行 |

LR两臂从同一个原5000**完整状态**分叉，保留初始权重、Adam各参数状态、RNG及固定输入一致。
只有block参数组倍率变化，embedding/pooling/unpool投影和Head维持原base LR。
分组必须逐参数名核验无重复、无遗漏，覆盖全部459组；验证warmup及退火期间倍率不会被scheduler覆盖。
这是新的训练配方变体，不能和“沿TreeLearn统一配方、只替换Backbone”的旧基线混写。
不同时修改weight decay、Head、GT、voxel/patch size、网络深度、序列化顺序或DropPath。

微型拟合先看：是否在两个已知完整输入上学到不同树的不同XY投票，
与零预测相比XY误差至少降低50%，且类内投票散度下降、数值/BN稳定。
50%是拟议工程筛选线，不是论文标准；200步不过只表示此预算内未过关，不能证明架构永远不可学。
同时报告与拟合集分开的固定crop表现，避免把记住样本当作泛化。
观察节点先做局部定点统计，不为每50步都跑一次完整L1W。
重复样本试验使用独立trial_step，不并入正式采样游标；后续正常阶段仍从原5000完整状态重新分叉，
不拿微型拟合产物冒充原计划的续训节点。

U-Net scratch若与已训练5000步的PTv3比较，只能作管线可学习性自检，不能算公平的Backbone结论。
正式架构对照必须另行安排E0-scratch/E1从头、相同样本计划和预算；
共同采用相同Head数值策略及资源边界，并区分相同训练配方与PTv3专用适配结果。

## 第三步：通过前两步后，再考虑500步有界阶段

建议届时最多从原5000完整状态推进到5500，设置5250、5500两个检查节点。
这是待决定的新预算，本计划不启动它；也不把12500 warmup终点当成必须支付的成本。

每个节点训练结束后，使用预先冻结的同一72个训练域crop做相同静态BN校准，
再用原195-tile、24树L1W协议评价。原5000校准结果作为共同对照。
校准只产生推理副本，不能用它代替完整训练状态续训。
比较TP/FP/FN、实例F1/P/R/Coverage，以及Seed层XY误差和投票可分性。
原生统计与校准统计同时留档，避免将训练收益与统计修正收益混合。

如果连续两个节点没有新增TP、Seed投票也没有改善，则停止该变体，保留失败证据。
若出现非有限数或统计失配再次扩大，先返回数值/分布诊断，不自动追加训练。
这些是成本止损规则，不是PTv3架构无效的统计证明。
预算实际耗时先按短段测量，不能直接套用U-Net或论文时间。

## 后续研究路线

当基部投票已能学会、但相邻树之间仍形成连接时，才有理由研究投票一致性或实例区分正则。
这属于后续独立结构/目标变体，需以TreeLearn错误分析为依据，不能马上叠加新模块掩盖未通过的基线。
若最终试验输入XYZ、法向或verticality，应明确改变了输入条件，设共同E0/E1对照；
不把常数输入直接当成已确认错误，也不把额外特征版本冒充原main Backbone-only实验。

本路线继续使用L1W作开发验证；Wytham不参与上述排查、校准或选择。
项目历史已经用过Wytham，后续不能称全新盲测；严格独立泛化结论还需未参与设计的数据。
当前不并行启动Local Split或其他历史分支的新训练。

## 本轮交付与状态

已完成：两篇论文关键章节和附录核对、作者2024配置固定版本核对、
原5000 optimizer的CPU只读检查、计划与证据文件。
未执行：本文件列出的前向/反向诊断、微型拟合、LR对照、E0自检、5250/5500续训与评价。
训练仍停在5000，原快照、冻结源和数据均不修改。

来源与事实/假设的结构化记录见 [sources_and_observations.json](evidence/sources_and_observations.json)，
执行顺序和预算状态见 [plan.json](evidence/plan.json)。
