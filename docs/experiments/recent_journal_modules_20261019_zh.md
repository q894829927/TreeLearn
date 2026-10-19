# 近期期刊点云模块：面向森林实例分割的组合候选

检索日期：2026-10-19。用户要求优先从近期三维点云期刊工作寻找可组合模块，以实际效果为目标。检索范围以 TPAMI、IJCV、ISPRS Journal of Photogrammetry and Remote Sensing、IEEE TGRS 为主，不将会议论文或只有预印本的工作混写为期刊论文。

本文依据可访问的期刊/作者机构页面、公开摘要和作者源码筛选；未取得的付费全文不宣称已阅读全文，也不据此声称完成论文复现。没有运行任何模型或实验。

## 1. 筛出的四项

| 工作 | 期刊及时间 | 值得借用的部分 | 本项目的接入点 | 代码核验与优先级 |
|---|---|---|---|---|
| DGLSS++：Toward Enhanced Representation Learning for Single-Source Domain Generalization in LiDAR Semantic Segmentation | TPAMI，2026 年 1 月在线，48(6)，2026 年 6 月卷期 | 不同稀疏度视图间的特征一致性，尤其是 GMSIFC 的混合体素过滤思想 | 主干训练目标；先做森林适配的简化变体 | 已核验期刊和作者机构记录；找到的 DGLSS 公开仓库 README 仍对应 CVPR 2023，未核验 ++ 完整实现。训练模块优先 |
| CoBa：Contextual boundary-aware network for semantic segmentation of complex land transportation point cloud scenes | ISPRS Journal，230，18–31，2025 年 12 月 | 高斯增强位置编码 | 主干输出到查询/掩码分支之间的局部几何残差分支 | 已读取核心源码；TensorFlow 实现，需移植为 PyTorch。结构模块优先 |
| P3Former：Position-Guided Point Cloud Panoptic Segmentation Transformer | IJCV，133，275–290，2025 年 1 月卷期；2024 年 7 月已在线 | Mixed-parameterized Positional Embedding、Position-Aware Segmentation、Masked Focal Attention | 实例解码中的位置表达和注意力更新 | 作者 PyTorch/MMDetection3D 仓库可访问；属于较早在线的期刊工作，不冒称 2026 新方法。解码器备选 |
| PointGrad：A Unified Spatially Varying Tensor Gradient Operator for Point Cloud Segmentation | TGRS，2026 | 方向梯度式局部几何算子 | 局部结构增强分支 | 作者页面确认期刊工作；所查公开仓库只有 README，未见模型源码。保留候选，第一轮不依赖它 |

一手来源：

- DGLSS++：[IEEE 期刊页](https://ieeexplore.ieee.org/document/11355705/)、[作者机构记录及摘要](https://pure.kaist.ac.kr/en/publications/toward-enhanced-representation-learning-for-single-source-domain-/)、[旧版作者代码](https://github.com/yskang05/DGLSS)。DOI：10.1109/TPAMI.2026.3654352。
- CoBa：[期刊页](https://www.sciencedirect.com/science/article/pii/S0924271625003582)、[作者仓库](https://github.com/Kange7/CoBa)、[核心文件](https://github.com/Kange7/CoBa/blob/main/Coba/model/coba.py)。DOI：10.1016/j.isprsjprs.2025.09.006。
- P3Former：[IJCV 期刊页](https://link.springer.com/article/10.1007/s11263-024-02162-z)、[当前作者仓库](https://github.com/InternRobotics/P3Former)。DOI：10.1007/s11263-024-02162-z。
- PointGrad：[作者发表列表](https://baorina.github.io/)、[论文 DOI](https://doi.org/10.1109/TGRS.2026.3686390)、[所查仓库](https://github.com/baorina/PointGradNet)。仓库存在不等于模型代码已经发布。

## 2. 第一优先组合：局部结构加密度鲁棒性

**A = 完整 ForestFormer3D；B = CoBa 风格高斯局部位置增强；C = DGLSS++ 思路的稀疏视图一致性训练。**

这是一项工程组合建议：B 是结构模块，C 是训练模块。目标分别是改善局部结构表示，以及减少特征对点密度的依赖。两项不占用同一个 query 接口，C 不增加推理时的视图分支。

与[前案：混合查询加查询竞争](abc_forest_query_plan_20261019_zh.md)共用完整基线 A；前案保留为查询模块候选。本次新增期刊模块，不把所有候选一次装进模型。若已有可运行的 SaTv2 完整底座，B/C 同样可以接入其主干特征；当前工程起点继续使用已核验接口的 ForestFormer3D。

推荐这组组合的理由来自本项目已观察到的跨森林表现差异与复杂树冠分离需求，但这些现象不能证明密度变化就是失败的唯一原因，也不能证明 B/C 在森林数据上必然有效。

## 3. B 的具体移植

作者核心源码提供了相对邻域坐标的高斯编码，例如每个轴的 `exp(-delta_axis**2 / (2*sigma**2))`，并与几何信息融合。接入本项目时只借用这一局部编码思想，不移植整个 TensorFlow 网络。

拟采用的 PyTorch 分支接口：

```text
输入：本块体素坐标 X[N,3]、主干特征 F[N,D]
邻域：带数量上限的局部邻居索引
几何编码：相对坐标、距离、若干尺度的轴向高斯响应
聚合：投影后与邻居特征联合聚合
输出：F_enhanced = F + gate * local_geometry_branch(X,F)
```

`gate` 从 0 开始，输出维度仍为 D；先在一个尺度接入，原查询选择、掩码预测和质量分支读取增强后的特征。局部坐标使用共同块坐标系，距离尺度与体素大小对应。多尺度响应和门控是本项目拟采用的变体，不是作者代码的逐项复现。

训练仍由 A 的实例/语义目标优化。邻域边分块聚合，避免一次保存整个森林的 `N × K × D` 张量；保持原有流式块处理方式。

**不直接照搬语义边界损失。** 两棵相邻树常属于同一语义类别，树/非树或 wood/leaf 边界不能代替树实例边界。如果以后增加边界监督，需要使用已有实例 ID 构造已知标签点对，未知实例点对排除。第一版只接几何分支，避免同时改结构和监督。

## 4. C 的具体移植

期刊工作的核心动机是源域训练下的稀疏/稠密变化一致性；原任务是道路 LiDAR 语义分割。森林第一版借鉴其中的特征一致性思路，不冒称完整 GMSIFC/LSCC 实现。

1. 从同一个源域森林块生成原密度视图和随机稀疏视图，保存原始点 ID 与体素映射。坐标变换和归一化参考保持一致。
2. 共享 A+B 的模型参数分别前向，沿共同物理点或可靠体素对应取得两份特征。只在相同来源的有效对应上约束一致性，不跨任意邻点做最近邻强配对。
3. 用于一致性目标的原密度特征停止梯度，稀疏视图对齐其归一化特征；两个视图的原生任务损失照常反传，避免只学出常量特征。
4. 以逐步启用的小权重加入一致性项；特征投影头仅用于训练，推理时不运行第二视图。

示意目标：

```text
loss = task_loss(original_view)
     + task_loss(sparse_view)
     + lambda_consistency * feature_consistency(valid_correspondences)
```

森林适配尤其要过滤混合体素：除了未知标签，也检查是否混入多个树实例，避免强迫相邻两树的混合特征一致。原论文的语义类别过滤不能自动解决同类不同树的混合问题。

第一版不直接约束两个视图的第 q 个 query 掩码相等，因为查询采样后顺序及身份可能变化。也不首先移植多语义类别的原型相关性损失：树/非树类别有限，类别相关性本身不足以区分邻树实例。

C 只使用源域训练块，不需要 Wytham GT 或目标森林参与训练。使用目标域无标签点云另做适配是不同设置。训练需要第二次前向，额外计算和显存应记录；推理仍是一套模型、一次前向。

## 5. 两个备选模块怎么用

P3Former 的 PA-Seg/MFA 更接近实例解码问题，可作为后续替换 decoder 位置表达的候选。森林版本要保留高树的竖直跨度，并避免固定空间窗口截断偏冠或倾斜树。ForestFormer3D 已有掩码注意力，单纯再加普通 attention 不等于移植了 MFA。先读清其具体更新和位置参数，再独立与 A 对比。

PointGrad 从题目和作者发表记录看与局部几何学习相关，但当前源码发布状态不足以支持“直接拿模块接入”。若以后根据全文重实现，应明确实现来源和与论文的差别；不因此拖延已有可读代码的 GPE 分支。

另检索到多模态预训练、时序点云、网格重建等期刊方向。它们依赖本项目不一定具有的图像、时间序列或网格，未作为第一轮模块推荐。只因为发表在目标期刊，不代表适合移植到本任务。

## 6. 第一轮实验

保持一个基线 A，按 A、A+B、A+C、A+B+C 进行四组实验。数据划分、初始化来源、主任务损失、查询预算和输出规则保持一致，记录 C 增加的训练计算。先用一个随机种子判断方向，有效后再补重复种子。

源域看 F1、TP/FN、实例重复和按 GT 树等权的最佳 IoU Coverage；Wytham 看相同规则能否改善原有的召回与覆盖退化。它已经参与本项目开发分析，不能把这轮结果重新当成全新的独立测试。论文的语义 mIoU 或别的协议下的覆盖率不直接与当前实例指标比较。

本次仅核验资料和公开源码、保存设计文档。没有改训练代码，没有在本地训练、推理或测试。完整可执行的模块移植与服务器前台运行入口属于后续实现交付，当前不提供不存在的命令。
