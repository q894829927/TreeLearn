# PTv3 基部投票定点诊断与受限可学习性试验

执行依据为已批准的[论文后续计划](../ptv3_paper_plan_v1/README_zh.md)。
本目录区分零更新诊断 D0、重复固定输入的本地试验 D1/D2，以及原正式采样计划。
**原正式状态仍为 update=5000；微型拟合的 trial_step 不并入正式游标。**
本轮已结束：D0、两臂各 200 次 D1、200 次 D2 均完成，拟合门槛未通过，T0 不进入。
最终决策见 [execution_completion.json](evidence/execution_completion.json)。

## 输入和保持条件

按既有 72 个训练域校准输入的元信息，选择 A1N、G3N、LG3 各自点数最小/最大的两个裁剪，
共 6 个，586,216—1,436,827 点。选择未使用预测结果。
复用原增强 seed、完整点坐标/GT/mask 及保存的体素输入和 point map；不重新生成数据池。
D0 三个模式使用同一 5000 参数：原始 running BN、既有顺序校准 BN、仅 BN 使用批统计。
三者均关闭 DropPath/dropout；批统计不写回 running buffer，所有 BN state 和随机状态都核对。
因此该比较隔离 BN，不能把它写成 DropPath 开关的单变量因果试验。

保留冻结的 main GT、3D Euclidean Offset loss、50×semantic CE、完整输入、0.1m voxel、
patch、通道、网络深度、序列化顺序及 native HDBSCAN/5NN 规则。
训练使用原 Backbone AMP / Head FP32 数值策略；D0 是全局 FP32 原生推理，
FlashAttention 内部半精度仍按冻结实现执行。
Wytham 未参与这些输入、校准、诊断或选择。

## D0：映射通过，批统计也没有恢复 XY 收缩

6 个输入均保留全部原点和实例。offset mask 是 inner、已知语义、正实例的子集。
点→体素索引全部有效，所有点都在对应 0.1m cell 容差内；
`coords + offset_GT` 的逐实例 anchor 最大偏差仅 1.2e-6—3.7e-6 m。
这些检查没有发现符号、单位、监督 mask 或点序映射错误。
它们是所选输入上的检查，不能代替全部数据和未执行路径的证明。

|完整输入|零预测 XY m|校准 BN XY m|批统计 BN XY m|批统计相对零预测改善|模型 Seed|GT 近基部竖直点|
|---|---:|---:|---:|---:|---:|---:|
|A1N small|3.377|3.595|3.579|−6.0%|29|428|
|A1N large|3.336|3.344|3.336|约0.0%|313|1,494|
|G3N small|7.165|6.907|6.821|4.8%|736|3|
|G3N large|1.810|1.872|1.878|−3.8%|1|1,810|
|LG3 small|2.566|2.394|2.398|6.6%|680|550|
|LG3 large|3.495|3.255|3.288|5.9%|0|1,515|

XY 统计使用冻结训练监督的全部有效 inner 树点。GT 近基部竖直点按
`verticality>.6 && abs(offset_GT_z)<4` 定义，只用于事后分层；模型 Seed 仍使用预测语义和预测 Z。
各完整 35m 输入的监督区仅为 native 8m inner，可能只含少量可见树干或树冠片段；
尤其 G3N small 只有 3 个上述 GT 近基部点，不能用该小分层估计普遍树干性能。
这些训练裁剪不是固定 L1W 的 24 棵完整树，表中也没有新实例 F1。

校准 BN 的同树投票 RMS 与零位移对照几乎相同（逐例宏平均约 0.917/0.910、
1.305/1.303、1.176/1.168、1.307/1.306、1.162/1.168、1.293/1.295 m）。
批统计的情况相近。校准和批统计缓解了原统计偏移，却没有形成充分的 XY 学习证据。
不能仅用整体 XY 绝对值或 loss 下降判断实例有效。

末级 `backbone.dec.dec0.block1` 的输出/输入 RMS 比为 406.7—420.3。
例如 A1N small：输入 RMS 7.22，attention 输出 RMS 262，MLP 输出 RMS 2773，
整个 block 输出 RMS 3019；最终平衡标签输出特征的有效协方差秩约 1.00—1.04。
强共模和近一维主导支持检查优化/表示尺度，但不证明全部几何信息已丢失，也不单独证明 LR 是根因。
完整 CPE/attention/MLP、pool/unpool、patch 跨度和少数实例计数保留在运行 JSON 中。
池化越深，多树混合组比例增大；这是分辨率变化的测量，不能直接当作实现错误。
特征和 patch 统计只抽取输出样本，不减少任何前向输入点。
报告的 `votes_shared_axes.png` 从缓存预测在 CPU 重绘，统一米制坐标轴、逐 GT 实例独立配色和图例；
原运行图仍保留在运行目录。重绘没有增加模型前向。

两个案例分别做 GT semantic、GT XY、GT Z 替换，共 6 次缓存数组 native 聚类诊断。
单项替换都没有产生有效簇；未知 GT 不替换，聚类阈值不变。
多个串联条件可能共同阻断结果，因此这不构成严格性能上界，也不能据此判定某一个组件正确。

## 梯度归因与 D1 对照规则

在两个拟合输入上分别独立反向 50×semantic 和 3D Offset，共 4 次，无 optimizer 更新。
每次恢复原 net 和 RNG，在可丢弃模型状态中使用 native AMP、DropPath 和训练 BN。
所有梯度有限。三维 Euclidean 输出梯度的逐轴 RMS 为：

|输入|X|Y|Z|末级 block semantic / Offset 梯度范数|
|---|---:|---:|---:|---:|
|A1N small|0.523|0.622|0.582|8.14 / 18.36|
|G3N small|0.545|0.700|0.462|65.98 / 16.87|

没有看到 Z 全局主导梯度；不以此删除 Z 监督或更改 loss。
两种 loss 在不同输入和模块的相对贡献不同，梯度报告以 block/pool/head 分组保存。
这些原训练模式结果与 D0 的 FP32、DropPath-off 推理条件不同，不能把差异全归因于 DropPath。

D1 两臂均从原 5000 完整快照分叉，固定 A1N small + G3N small 两个完整输入，
每个 trial_step 做两次 microbatch 累积，并按各自有效语义/Offset 点数归一化。
原 Adam 每参数 state、模型、scheduler、scaler 和初始 RNG 保留。
459 个参数张量逐名覆盖，无重复和遗漏；仅 `.block` 参数使用 1.0 或 0.1 的 base LR 倍率，
embedding/pool/unpool/Head 的 base LR 相同。本地 LR 时钟对应 epoch21，
base=0.0012658，低倍率 block=0.00012658；既有 cosine state 加显式相对倍率适配，
warmup 和退火节点均核验，不能让 scheduler 覆盖倍率。

每臂最多 200 次尝试，检查 0/50/100/200；不推进正式 sampler，
不把重复样本权重冒充正常采样计划的续训结果。
局部检查使用批 BN 且 DropPath off，保留 BN/RNG；
AMP 的 sparse eval kernel 在此环境不可用，检查时仅 SparseConv 保持训练 kernel，其余随机模块仍关闭。
这是一致的局部诊断口径，不替代正式 FP32 L1W 评价。
两个 large 输入仅作未拟合的固定局部对照，不能称独立泛化评价集。

预设工程门槛：两个拟合输入都比零预测 XY 误差至少低 50%，同树投票 RMS 下降，且数值状态稳定。
50% 不是论文标准；200 次不过只表示本预算内未通过。
0.1 臂从已有 5000 状态恢复，能测试短期修复，不能判断从头使用该倍率是否会避免既有退化。

## D1/D2 完成结果及止损决定

下表是第 200 次本地尝试后的批统计检查。两个拟合输入的零预测 XY 误差分别为 3.377、7.165 m。

|试验|A1N small XY m / 改善|G3N small XY m / 改善|两者同时通过|新跳步|
|---|---:|---:|---|---:|
|PTv3，block 倍率1.0|3.110 / 7.91%|6.773 / 5.47%|否|0|
|PTv3，block 倍率0.1|3.043 / 9.91%|6.718 / 6.25%|否|0|
|U-Net scratch 管线自检|3.560 / −5.43%|6.339 / 11.53%|否|0|

PTv3 两臂的同树投票 RMS 分别为 0.916/1.168 和 0.921/1.171 m，
零预测对照为 0.910/1.168 m，未形成有意义的收缩。
未拟合 A1N large / G3N large 的 XY 改善分别为：倍率1.0的 +17.65%/−26.50%，
倍率0.1的 +25.53%/−34.67%；局部正向变化不一致，不能宣称普遍改善。
0.1 臂的训练梯度范数后期较平稳，但工程门槛仍失败，不将它选为有效模型。

U-Net 从 seed42 随机初始化，原 main 七级结构、两层 Head、3D GT/loss 和相同两个完整增强输入均保留，
Head 同样采用 FP32；体素化使用 native U-Net 路径，不限点、不改变固定 spatial shape。
它在首次完整累积训练时通过显存门槛，并完成 200 次成功更新。
scratch 使用原配方的 epoch1 LR=0.0000698；已训练 PTv3 使用 epoch21 LR=0.0012658。
两者初始化、LR 时钟和训练历史不同，不能拿此试验比较架构优劣。
U-Net 也未通过短拟合门槛，说明当前预算/配方下的 Offset 可学习性尚未建立，
不能把 PTv3 的失败单独归因为其 Backbone。

|试验|200 次循环及节点检查耗时 s|训练 peak allocated / reserved GiB|保存节点 BN count|保存节点 Adam step|
|---|---:|---:|---|---|
|PTv3 倍率1.0|231.52|5.82 / 17.20|10100 / 10200 / 10400|5041 / 5091 / 5191|
|PTv3 倍率0.1|235.86|5.82 / 16.92|10100 / 10200 / 10400|5041 / 5091 / 5191|
|U-Net scratch|63.66|3.60 / 5.64|100 / 200 / 400|50 / 100 / 200|

三项合计 600 次本地 optimizer 更新、1200 次累积训练前向/反向，另有上述 4 次诊断反向。
本地节点检查 4 输入×4 节点×3 项=48 次前向，不属于 D0 的 18 次普通前向预算。
所有训练输入点完整保留；三项均无新 OOM、非有限梯度、AMP 跳步或 BN 异常。
CPU 逐份检查 50/100/200 的 net 和 Adam 所有浮点状态有限，BN 次数、Adam step 均符合预期。
这里的时间包含本地节点检查和保存，不包含 Python 导入、初始输入装载/检查等启动成本。

**T0 的可学习性条件未满足，本轮停止，不执行 5000→5500，也不新增 L1W 校准或评价。**
原正式 checkpoint、sampler 和阶段文件的 hash 没有变化；三个冻结源 git tree 均 clean。
此次没有提供新的正式实例分割 F1，旧 5000 校准后的 2/24 TP 仍是最近一次该协议的结果。

建议下一项先增加零更新的“初始化→已有节点”参数/表示演化审计：
定位末级 decoder 的 attention/MLP 共模放大何时形成，检查 LN/投影/Adam 更新尺度是否同步异常，
并核对两个固定样本中远离 inner 的基部锚点及长 Z 目标对短期拟合的影响。
然后再设计两骨干从头、相同本地 LR 时钟的有限拟合对照，或独立的 PTv3 从头倍率对照。
这是后续建议，不在本轮继续追加更新；不同时改变 GT、loss、输入特征或网络以混淆原因。
200 次失败并不证明目标不可学、架构不可用或 0.1 倍率从头训练无效。

## 故障与成本留档

D0 首次在 pooling 诊断钩子处退出：`Point/addict` 复制嵌套对象后身份变化。
追踪改用不变网格张量 storage 标识。已完成的第一个原 BN 前向被复用：
合计 18 次完整成功前向，另有 1 次中途退出的诊断前向，未发生模型更新。
18 次成功前向计时合计约 8.72 s；进程总耗时还包括导入、CPU 完整统计和文件 I/O。

0.1 臂第一次预检在浮点倍率精确相等处退出，尚无前向/反向或参数更新；
改为 `rel_tol=1e-14` 的严格容差。两臂执行源码差异只有该 CPU 检查及 math import，
实际训练段代码相同，初始 RNG、输入、父快照与指标代码逐项核对。
失败目录、完整调用栈、执行源码和所有成功结果均保留，不覆盖原始实验。

## 证据与复现入口

- [D0 比较 CSV](evidence/d0_comparison.csv)、[逐轴/角色完整比较入口](evidence/runtime_index.json)。
- [feature/patch/pooling 摘要](evidence/feature_geometry.json)。
- [D1/D2 节点表](evidence/trial_nodes.csv)、[受限试验汇总](evidence/trials.json)。
- [两臂起点及训练代码核对](evidence/matched_arm_verification.json)、[冻结源核对](evidence/frozen_sources_verification.json)。
- 大型输出：原运行根 `diagnosis/vote_diagnosis_v1/attempt_v2/`、`diagnosis/vote_microfit_v1/`。
  包含每例三个模式的 JSON/完整 inner 预测、投票图、full traceback 和每 50/100/200 的完整本地 trial 快照。
- [D0 工具](../../../tools/experiments/ptv3_vote_diagnosis_v1.py)、[受限试验工具](../../../tools/experiments/ptv3_vote_trial_v1.py)、
  [CPU 报告索引](../../../tools/experiments/ptv3_vote_report_v1.py)、[缓存投票重绘](../../../tools/experiments/ptv3_vote_plot_v1.py)。

后续是否进入 T0 只由预设门槛决定；不因 loss 下降自动推进到 5500、12500 或 250000。
原始 5000 快照、冻结源、校准推理副本、固定 L1W 协议及原阶段状态保持原样。
