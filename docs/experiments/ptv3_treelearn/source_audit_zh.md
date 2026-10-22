# 基于公开 TreeLearn main 的 PTv3 骨干替换与跨森林泛化对照实验

冻结来源：TreeLearn `fd240ce7caa4c444fe3418aca454dc578bc557d4`，从实时获取并经 `git ls-remote` 核验的 `origin/main` 建立 `codex/ptv3-treelearn-baseline`。原工作目录和历史分支未切换、合并或重置。独立 worktree 位于 `/home/szu-stu/.codex/worktrees/ptv3-treelearn-baseline/TreeLearn`。

PTv3 来源为 [Pointcept 官方 detached 实现](https://github.com/Pointcept/PointTransformerV3/tree/3229e9b7de1770c8ad17c316f8e349982de509f8)，commit `3229e9b7de1770c8ad17c316f8e349982de509f8`，MIT 许可证。`vendor/model.py` 与 `vendor/serialization/*.py` 按字节保留上游实现，许可证随代码交付。新增的适配、训练管理和极端输入处理位于独立实验模块；不修改 main 原有模型、数据集、工具或默认配置文件。

## main 的实际行为

| 模块 | 输入、输出与冻结行为 |
| --- | --- |
| 原始点云读取 | `load_data()` 将 LAS/LAZ 的 `treeID` 与 `classification` 解释为实例标签；class 1/2 是非树，treeID=0 且不属于非树的点是 -1。正标签不按高度重标注。 |
| 森林体素下采样 | `data_preparation.voxelize()` 将 XYZ 四舍五入至 0.01m 后用 Open3D 0.1m 网格取点均值；标签取 trace 中第一个原始点，非多数投票。 |
| Verticality | jakteristics，半径 0.6m，NaN 用同一特征的全林均值替换。仅供实例 seed 筛选，首轮不作为可学习输入。SOR 与半径过滤均关闭。 |
| 训练裁剪 | 25,000 个 35m×35m 候选旋转窗口；1m occupancy，100,000 点 occupancy 抽样，9 邻域填充 0.9，候选 occupancy 严格大于 0.45。按森林 occupied cells 分配数量并修正舍入余量。候选网格边长为 `int(sqrt(max(25000,5*n_plot)))`，角度 uniform(0,2π) 后保留 2 位小数。 |
| 推理 tiles | inner=8m，outer=13.5m，stride=0.5；实际完整上下文约 35m。仅保留内区有点的 tile。 |
| 增强顺序 | 首先以概率 0.25 逐点 jitter（σ=0.1m，clip=0.2m）；然后各项以概率 0.5 执行矩阵增强。XY scale 独立 uniform(0.8,1.2)，Z scale uniform(0.95,1.05)；矩阵扰动 N(0,0.1)；X 对角项随机 ±1；Z 轴 uniform(0,2π) 旋转。 |
| Base Offset GT | 在增强后完整 crop 上逐实例处理：点数>11 时直接使用 `np.partition(z,10)[3]`，否则 min(z)；取 z≤参考高度+0.5m 的 XYZ 均值为 anchor，GT=anchor−xyz。该 partition 前缀并未排序，不能替换为排序后的“第三/第四低点”。 |
| 监督 masks | 增强和 GT 之后计算内区 `max(abs(x),abs(y))≤4m`。semantic mask=内区且实例标签≠-1；offset mask 再要求树语义与 `mask_valid_offset`。ignore 实例也可能被临时计算 anchor，但永不进入有效监督。 |
| batch 张量 | coords `[N,3]` float32，input_feats `[N,1]`，batch_ids `[N]`，semantic/instance labels `[N]`，offset_labels `[N,3]`，masks `[N]`，centers `[N,3]`。语义 tree=0、non-tree=1。 |
| 模型体素化 | `tree_learn.model.tree_learn.voxelize()` 每样本以最小坐标为网格原点，0.1m，最多 3 点用于体素特征均值；保留每个输入点到体素的 map。原实现 `use_coords=false,use_feats=false` 时实际为 **4 通道全 1**，而非删去坐标通道。 |
| Sparse U-Net | 输入卷积 4→32，七级通道 `[32,64,96,128,160,192,224]`，kernel=3，每级两个 ResidualBlock，原始 down/inverse-conv/skip decoder，最终 `[V,32]`。训练和推理默认 spatial_shape `[500,500,1000]`。 |
| 两个 Head | 先按体素逆映射恢复 `[N,C]`，然后原版 `MLP(C,2,num_layers=2)` 与 `MLP(C,3,num_layers=2)`；隐藏 Linear(C,C)+BatchNorm1d(eps=1e-4,momentum=0.1)+ReLU，末层 Linear。 |
| Loss | 50×有效点平均二类 CE，加有效 offset 点平均三维欧氏距离；非平方 L2、非 SmoothL1。空 mask 用预测之和×0 构造可反传零值。 |
| Ensemble | 内区点的坐标保留 2 位小数，按 XYZ groupby.mean；融合 **logits** 和 offsets，不是平均 softmax 概率。 |
| Seed | softmax(tree)≥0.5，verticality **>0.6**，abs(offset_z) **<4m**。 |
| 实例聚类 | 在 XY 的 coord+offset 上用 sklearn.cluster.HDBSCAN(min_cluster_size=50)，默认其他参数保持 sklearn 实现；tau_group=0.15 只用于可选 DBSCAN，非 HDBSCAN 半径。 |
| 未分配树点 | 在 **三维** coord+offset 空间对有效聚类参考点执行 5NN 分类投票。原版非树=0、未分配=-1、有效实例从 1 开始。 |
| 整林恢复 | main 的 hash 对齐和剩余点 5NN；坐标先整体居中，导出整林时加回精确均值。 |
| 评价 | 预测先 5NN 传播到 GT；Hungarian 匹配后用 **IoU>0.5** 过滤。未匹配预测只有在其对全部 GT 的 precision 之和≥0.5 时计 FP。GT 的 0 与 -1 都不作为匹配对象，但预测落在这些区域会影响点级 precision。 |

## main 训练循环与本次共同工程适配

main 每 epoch 对打乱后的 loader 前 250 个 batch 更新，batch=2，共 500 样本；1000 epochs 为 250,000 次更新尝试、500,000 次样本读取。main 还会从 loader 取出下一批后才 break，worker 也可预取，这不是额外训练更新。Scheduler 在每次更新前 `step(epoch)`，同一 epoch 学习率相同；第一 epoch LR 为 0.0000698。AdamW lr=0.003，weight_decay=0.001；t_initial=1000，lr_min=0.00005，warmup_lr_init=0.00001，warmup_t=50，其他 cycle 配置沿用 main。

main 对 **scaled 梯度**直接 clip，`grad_norm_clip: true` 在数值上是 max_norm=1；原 checkpoint 只含 net、optimizer、epoch，缺少 scheduler、scaler、RNG、sampler 状态。不能将本轮改进后的入口描述为 main 脚本逐操作复现。

所有 E0/E1 在本轮独立训练入口一致采用：

1. 先 GradScaler.unscale，再按 max_norm=1 裁剪；共同初始 AMP scale=1024，非有限梯度由 GradScaler 跳过并记录，不虚报有效更新。
2. 每个样本的 crop 索引与增强 seed 来自冻结计划；模型 dropout/DropPath/序列化的随机消耗不改变输入、GT 或 masks。
3. 保存 net、optimizer、scheduler、scaler、epoch/global_update、各 RNG、下一样本 cursor，以及代码、数据、配置、环境、计划哈希。恢复时身份不匹配报错。
4. 每 epoch 更新 last，每 20 epochs 留档；短程终点仅称 smoke_final/pilot_final。只有完整预算才写 final.pth。
5. 不读取 L1W/Wytham 进行训练、验证、早停或 checkpoint 选择，固定完整预算的最终 checkpoint 为主要结果。

GPU 体素化的重复调用实测会改变 voxel ID 和 point map 的编号顺序。资源适配接口对**三个训练组共同**按 batch/XYZ 排序体素，再同步变换逆映射；没有改变体素集合、特征值、点序或 GT。main 兼容配置仍可执行原始分配顺序。精确恢复模型/优化器/调度器/Scaler/RNG/游标与继续计算的位级一致性分别测试；当前 CUDA 路径不保证后者，测试预先规定绝对容差 2e-4 并保留最大差值。

另一个实际边界是 main 的固定 spatial_shape=[500,500,1000]：0.1m 网格的 XY 范围为 50m。G1N 短程测试的 200 个增强样本中，27 个超出这个范围，最大 XY Extent 为 65.75m、60.52m。**不修改原始 main 或兼容配置**。标为 `covered-grid` 的共同资源接口仅在输入超界时扩展 U-Net 空间尺寸，并对超出的轴按六级下采样的 64 倍数补齐；原本有效的输入保持 [500,500,1000]，已做前向一致性测试。PTv3 本身使用覆盖实际 grid 的上游 sparse shape。三个资源训练组统一采用覆盖全部输入的策略，记录每批实际 grid、使用 shape 和原 shape 之外的体素数。该修正不改变增强、GT、Masks、网络层或 Loss，不能称为 main 脚本逐操作复现。

### PTv3 接口与资源适配

PTv3 首轮采用标准宽度/深度：encoder depths=[2,2,2,6,2]，channels=[32,64,128,256,512]，heads=[2,4,8,16,32]；decoder depths=[2,2,2,2]，channels=[64,64,128,256]，heads=[4,4,8,16]；各层 patch=1024，order=[z,z-trans,hilbert,hilbert-trans]，stride=[2,2,2,2]，DropPath=0.3，RPE=false，CPE 保留上游稀疏卷积，FlashAttention=true。输出 `[V,64]`，沿用同样的两层 MLP 定义，宽度不同造成的 Head 参数差异单独记录。

复用 main 的体素化后，将同样的常数特征、XYZ grid、逐样本 voxel-center coord、batch 与累计 offset 交给 PTv3。序列化、shuffle/inverse、pooling/unpooling 都由上游处理；最后严格核对 grid/batch 顺序，并显式恢复输入点顺序（也覆盖 interleaved batch IDs）。确定性 marker、重复体素、多 batch 隔离测试用于验证对应关系。

资源方案均有独立 YAML，原始物理 batch=2 的入口保留。microbatch=1/accumulation=2 按每个 microbatch 的 semantic/off 有效点数分别加权；**BatchNorm 统计仍不同于物理 batch=2**，不声称逐操作等价。PTv3 的 tensor checkpointing 只重算 Attention/MLP，避免重算 BatchNorm 和 spconv，并检验梯度与 BN buffers；不缩减网络宽度、深度或 patch size。

单独的 `*_resource.yaml` 标记 common-input-resource-limited-400k-covered-grid：先完整增强和 main GT/masks，再用同一个 seed 抽取最多 400,000 点；保留 XY 极值与每棵可见树的底部点、实例点，余量均匀抽样。E0/E1 对同一计划使用完全相同的索引，每步记录前后点数、实例保留率和范围。该 PTv3 结果必须称为 **PTv3-resource-limited**，不能混入完整 main 输入的结果。资源策略不限制推理点数。

## 推理与评价的独立安全处理

正常路径调用 main 的 tile generation、pointwise inference、ensemble、get_instances、HDBSCAN、5NN、hash propagation 与评价函数。输入文件 staged 到每个独立 run，避免 main 的居中写文件行为覆盖旧数据；checkpoint 严格加载。保留 main 的推理精度执行方式：其 fp16 配置没有在 get_pointwise_preds 中启用 autocast；PTv3 FlashAttention 自身按上游执行 fp16 QKV。

main 的全林居中修复在写临时 NPZ 时只保存 XYZ，原始实例标签不进入推理 tiles；因此保存的 pointwise semantic_labels/offset_labels 不能直接用作真实评价 GT。本接口的诊断命令显式读取独立 benchmark GT、恢复精确坐标平移后再计算语义召回和 Offset 误差。

两个可选内存适配均不改变预测任务或实例规则。`--compact-pointwise` 在两个 Head 完成预测后省略**保存** backbone features（存为 [N,0]），logits/offset/verticality 完整保留；测试片与正常路径的最终实例标签逐点相同。`--stream-tiles` 逐 tile 保存，避免 main 同时保留所有完整上下文 tile 的 float64 数组；已与 main 逐数组、dtype、中心和 JSON 元数据比较一致。随机裁剪仍调用 main 的原始候选/occupancy/save 逻辑。为降低机械盘重复读写，新增裁剪使用无损 deflate level 1 NPZ，并在写入时计算 SHA256、直接审计原始输出数组；已验证数组值、dtype、字段顺序与原版 NPZ 一致。已完成的标准未压缩裁剪可以保留，所有实际文件分别记录 SHA256。文件容器和执行内存策略属于工程适配，不描述成 main 原有行为。

仅对原版会报错的输入增加共同处理：无树、seed 少于可形成聚类的数量、全 noise 时保留未分配=-1；无 query/reference 时跳过 5NN；参考点不足 5 时使用已有参考点；空输出可导出。LAS/LAZ 的 uint32 treeID 无法表示 -1，极端输入用 treeID=0/classification=0 保留未知含义并做 round-trip 测试。正常输入标签不变。评价增加无 GT/无预测、一棵 GT、分区点数<5 的安全分支，并明确记录；正常匹配和 Commission 规则不变。

额外诊断的 Offset GT 使用 main getOffset 对**整片可见 GT**计算，和训练的增强 crop anchor 区分；按 nearest point 对齐并记录距离分布。尺度均标为 Z Extent，不称地面归一化树高。GT 不进入预测的 seed、聚类或采样。

## 数据与已发布模型的解释边界

18 块训练森林恰为任务指定名单；逐文件 SHA256、点数、实例数及 Z Extent 等位于 `source_manifest.json`。已通过官方 Dataverse 的发布 MD5 核验，合计 6,665 个标注实例。源文件 SHA256 是运行身份，MD5 仅用于和发布平台的原始校验值比对。

L1W 输入为 85,289,646 点、200 个正标签；评价 GT 为 20,434,648 点、**156** 个完整标注实例，两者坐标边界一致。评价 GT 的未知区域和边界按 main 的标签规则处理，不额外裁出论文中的其他区域。Wytham 本地输入和 GT 均为 54,592,456 点、877 个正标签，具有相同坐标范围；具体标签按本地文件和 main 读取结果定义。Wytham 为**跨域开发评价**，不称首次盲测。

HAIS 来源为 main training guide 的 SoftGroup Google Drive 链接，已独立重下并对 SHA256；详细层加载报告区分成功、缺失、形状不符和多余参数，并对全部 U-Net/output 必需张量作 100% 覆盖门禁。官方 `with_small_20241213` 单独命名并核验发布 MD5。默认 `20241213` 单独下载/核验，不用 with_small 静默代替。

E2 只作成熟模型参考。公开权重描述没有提供足以排除 L1W/Wytham 历史训练重叠的逐样本训练清单，因此该重叠状态记为**未证实排除**，不能将 E2 与从头训练 E0/E1 的差异归因于 Backbone。
