# 有具体源码且容易移植的期刊点云模块

筛选日期：2026-10-19。本版按用户最新要求重筛：没有对应模型源码的跳过；跨框架重写、需要额外模态/标注、依赖新专用 CUDA 算子或大幅重做数据流程的跳过。目标是先得到有效的组合架构。

本版替代此前的 CoBa + DGLSS++ 推荐。只做了期刊信息、作者仓库和具体类/调用位置的静态核验，没有安装依赖，没有运行任何模型、训练、推理或测试。以下“容易移植”指所选子模块，而不是整个原仓库可以直接运行。

## 1. 保留的模块

| 优先级 | 来源与准确时间 | 已读取的具体源码 | 接入位置与改动量 |
|---|---|---|---|
| 1 | TripleMixer，IEEE TIP 34:7712–7727，2025；2024 预印本、2025 更新与接收 | `network/geometry.py` 的 `GeoMixer`，约 46 行类定义；`network/model.py` 确认实际调用 | 主干输出进入实例解码器前；提供特征和局部邻居索引，增加维度转换/残差包装 |
| 2 | P3Former，IJCV 133:275–290，2025 年 1 月卷期；2024-07-27 已在线 | `p3former/decode_heads/p3former_head.py` 的 `_Masked_Focal_Attention`，以及 `_Transformer_Decoder` 的调用 | 替换一层或数层 query 更新；移除注册器，将 LN/GELU 构建器改成原生 PyTorch |
| 3，轻量备选 | GEPT-Net，ISPRS Journal 221:20–43，2025 年 3 月 | `models/GEPT_Net/gept_net_base.py` 的 `ChannelAttention`；主网络 forward 确认调用 | 主干特征通道门控；十几行主体，维度转换即可；不是完整 GECA 复现 |
| 同源轻量备选 | TripleMixer，TIP 2025，同第一项 | `network/mixer.py` 的 `ChaMixer` | 原生通道 MLP + 可学习逐通道缩放 + 残差；若只需最轻模块可先试，不与主干已有 FFN 重复堆叠 |

这些模块分别来自三篇论文，TripleMixer 的两个子模块不能算两篇不同研究。近期与容易移植无法同时满足时，保留准确日期，不拿期刊卷期把早期工作说成全新的方法。

一手来源与直接代码：

- TripleMixer：[作者仓库及期刊引用](https://github.com/Grandzxw/TripleMixer)、[TIP 论文 DOI](https://doi.org/10.1109/TIP.2025.3629047)、[GeoMixer 类](https://github.com/Grandzxw/TripleMixer/blob/main/network/geometry.py#L19)、[实际调用](https://github.com/Grandzxw/TripleMixer/blob/main/network/model.py#L34)、[ChaMixer 类](https://github.com/Grandzxw/TripleMixer/blob/main/network/mixer.py#L82)、[作者预印本](https://arxiv.org/abs/2408.13802)。源码以作者 Grandzxw 仓库为准，不依赖镜像仓库。
- P3Former：[IJCV 期刊页面](https://link.springer.com/article/10.1007/s11263-024-02162-z)、[MFA 类](https://github.com/InternRobotics/P3Former/blob/main/p3former/decode_heads/p3former_head.py#L16)、[原解码器配置和调用](https://github.com/InternRobotics/P3Former/blob/main/p3former/decode_heads/p3former_head.py#L124)。
- GEPT-Net：[作者机构期刊记录](https://dare.uva.nl/id/65fe5641-5918-45f8-a4ca-4eedfefbd26d)、[ChannelAttention 类](https://github.com/jdjiang312/GEPT-Net/blob/main/models/GEPT_Net/gept_net_base.py#L529)、[实际调用](https://github.com/jdjiang312/GEPT-Net/blob/main/models/GEPT_Net/gept_net_base.py#L741)。DOI：10.1016/j.isprsjprs.2025.01.032。

## 2. 第一优先：TripleMixer 的 GeoMixer

实际签名：

```text
GeoMixer(channels_in, channels_out)
forward(x[B,Cin,N], neighbors[B,K,N]) -> [B,Cout,N]
```

源码用 `BatchNorm`、1×1 卷积、`torch.gather`、邻居特征差、邻居维 softmax 和加权求和。它把逐点投影与局部聚合分支拼接后再投影；不是一个只对每个点独立运行的 MLP。

原始类不在内部查询邻居。作者数据入口通过 `scipy.spatial.cKDTree` 构造坐标近邻，源码第一行邻居被当作自身跳过。因此适配时应显式把自身放在第 0 行，至少再有一个有效邻居；不能依赖重复坐标时 KDTree 恰好返回的顺序。少点块和 padding 要有明确处理，不能用无效索引参与聚合。

森林接法：

```text
块内坐标 X[N,3] + 可训练主干特征 F[N,D]
  -> 坐标局部近邻索引
  -> GeoMixer(F 的维度转换, neighbors)
  -> 转回 [N,D]
  -> 残差接回 F
  -> 原实例解码器
```

只借用特征聚合类，在森林实例损失下联合训练。它接收可配置通道数，不强制需要反射强度；给它已有学习特征是本项目适配，不是原天气去噪模型的完整复现。不能把天气去噪权重直接用于删除枝叶点。

该类本身不需要新 CUDA 扩展；近邻使用现有空间索引或 CPU KDTree，不能对整片森林建立 N×N 距离矩阵。先接一层、一个尺度、有限邻居，例如 8/16；优先复用受控点数的块/粗尺度输出。原实现会创建邻居特征张量，其存储随 N×K×D 增长，不能把“模块少”当成“整林内存恒定”。训练时 BatchNorm 的统计依赖批次/点集，任意拆分聚合会改变训练行为，不能未经验证宣称分块前后完全相同。

其适用性是设计推断：显式处理局部特征差，可能帮助相邻树冠和细枝结构的表征。原论文是恶劣天气去噪，不能以其去噪收益证明森林实例 F1 会提高，也不把它直接描述成解决重复查询的模块。

## 3. 第二优先：P3Former 的 Masked Focal Attention

实际签名及输出：

```text
forward(queries[Q,D], features[N,D], masks[Q,N]) -> [Q,1,D]
```

原解码器把 `in_channels`、`feat_channels`、`out_channels` 都设为相同的 D；不能直接照抄类定义里不相等的默认通道组合。接入不同解码器时，将输出的长度为 1 的轴按原接口处理。

源码的关键行为是：

1. 用上一轮掩码 logits 经 sigmoid 后大于 0.5 的区域，汇总对应点特征；原代码使用求和，不是均值。
2. 从 query 生成动态参数，与区域汇总特征交互。
3. 用 input/update 两组门控合成新 query，再投影、归一化和激活。

这和普通 masked multihead cross-attention 不同，不能把 A 已有的掩码注意力重新命名成 MFA。所选类使用的归一化、激活构建器可分别替换成 `nn.LayerNorm`、`nn.GELU`，去掉模型注册装饰器；不需要搬入整个 MMDetection3D、Cylinder3D 主干和稀疏卷积。主模型 A 仍使用自己的既有依赖。

森林接法：保留第一层原有 cross-attention，以已有中间预测产生 mask logits；先把一层后续 query 更新替换成 MFA，保留原 self-attention、FFN、质量预测和实例损失。不要同时移植 PA-Seg、柱坐标输入和匹配策略。

需要保留的实现注意点：掩码 hard threshold 不对 mask logits 传梯度；空掩码需要回退到原 query 或原更新；N 较大时不能额外复制多份 Q×N 掩码。以上属于接入细节。若改用 soft mask 或按点数求平均，必须明确记为本项目变体，不能说是原始 MFA。

目标是让 query 更新读取自己目前解释的区域，检验掩码质量和树召回能否改善。它没有显式的“重复查询为负样本”机制，不声称直接替代 CompetitorFormer 的竞争模块。

## 4. 更轻的备选怎么取舍

GEPT-Net 的 `ChannelAttention` 是 `AdaptiveAvgPool1d -> Linear -> ReLU -> Linear -> Sigmoid` 的 SE 类门控。作者实际把 `point.feat[N,C]` 变成 `[N,C,1]` 调用，因此其公开实现的这一处是逐点通道门控，不是跨整片森林的全局特征池化。

只提取该类，不需要移植其 Pointcept/PTv3 主干，也不强制计算 FPFH 或补反射强度。这样得到的是普通通道注意力的森林适配；不能包装成完整 GECA 或独立提出的新机制。它适合作为低成本对照，优先级低于局部几何聚合和实例查询更新。GEPT 源码中的 RPE 也与 PTv3 结构同源，不另当成一项新贡献。

TripleMixer 的 `ChaMixer` 接受 `[B,C,N]` 并返回同形状，核心是 `tokens + DropPath(scale(MLP(norm(tokens))))`。选择 BatchNorm 路径即可避开原文件 LayerNorm 路径的额外维度约定；拆类时不导入它所在文件的 wavelet 分支。它与常规 FFN 功能相近，先检查基线中有无重复，再决定是否用作轻量对照。

## 5. 已跳过的候选及具体原因

| 工作/部分 | 跳过原因 |
|---|---|
| CoBa | 已读核心实现为 TensorFlow；按用户要求不再做跨框架重写 |
| DGLSS++ | 本次核查找到的作者仓库仍是 DGLSS 早期实现，未取得对应 ++ 模型/训练模块源码 |
| PointGrad | 所查仓库未见可用模型实现，不能仅按论文题目推荐移植 |
| BGPSeg 的 BGFE | `boundaryops` 专用 CUDA 邻域算子、边图和边界预测/监督一起参与，非独立即插模块 |
| PointGST 的 PCSA | forward 需要全局/局部图谱基 U、sub_U 和索引，配套谱分解流程超出简单特征插入 |
| InvSpaceNet/PointSegBase | 主要依赖语义类别原型/类别层级，不能直接把类别不平衡或语义层级当成不同树实例的问题 |
| LPFP/PCLN 完整主干 | 读到额外点云内核和多级采样聚合依赖，当前不为小模块引入完整新主干 |
| TripleMixer 的 FreMixer | 需要多个二维平面的格点映射和小波处理，第一轮不增加该数据流程 |
| GEPT-Net 完整主干/完整 GECA | 涉及 FPFH、强度及完整稀疏主干流程；只保留确实独立的通道类作为备选 |

跳过表示不进入当前易移植清单，不是论文无效，也不断言将来永远没有对应代码。BGPSeg 等是读到具体实现后因成本排除，不能继续沿用此前“可能合适”的初步判断。

## 6. 当前建议的 A+B+C

**A = 完整 ForestFormer3D；B = TripleMixer GeoMixer；C = P3Former MFA。**

```text
点云块 -> A 的可训练主干 -> B 局部特征聚合
       -> A 原有查询初始化/首层解码
       -> C 区域汇总与门控查询更新
       -> 原 self-attention/FFN、整树 mask 与质量输出
```

A 的作者代码已经有 `[N,D]` 的 `inst_feats`、`[Q,D]` 的 queries 及迭代掩码预测，接口适合接入所选子模块：[ForestFormer3D 解码器](https://github.com/SmartForest-no/ForestFormer3D/blob/main/oneformer3d/query_decoder.py#L286)。实际实现仍需检查当前训练配置使用哪个 decoder 类，以及特征/掩码是否处于相同的点或超点索引空间，不能按文件里第一个类盲接。

这个组合中 B 改局部特征，C 改查询更新，接口分工明确。完整 A 的建立工作独立于“小模块容易移植”；现有冻结 TreeLearn 特征上的 v1 不视为 A，也不应继续用它替代完整主干联合训练。

第一轮按 A、A+B、A+C、A+B+C 运行，同一数据与输出规则先比较单种子实际方向。已有源域基线高，重点同时看 Wytham 的 TP/FN、F1 与 Coverage，而不是只看减少 FP。效果改善后再补细分树尺寸/密度分析与多种子。Wytham 已用于开发分析，这轮按已使用的外部森林报告，不重新称为全新盲测。

下一次实现从完整 A 的服务器可运行入口开始，随后加可开关的 B，再加 C；提供终端前台、实时输出的命令。本文没有模型代码修改，不提供尚不存在的实验入口，也未在本地运行实验。
