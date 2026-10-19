# ForestABC v1：先跑完整端到端组合

日期：2026-10-19。按用户最新要求，第一轮直接训练完整 A+B+C，整体有效后再做模块消融。代码已实现；本地没有安装新模型依赖，也没有运行模型、训练、推理或单元测试。服务器首轮先执行运行检查，实际性能以服务器结果为准。

## 实际交付的网络

```text
坐标（已有 0.1 m 对齐点集）
  -> 0.2 m 输入体素，坐标特征
  -> 完整五级 SpConv U-Net（32/64/96/128/160 通道）
  -> TripleMixer GeoMixer 局部特征聚合 + 残差
  -> 二分类语义 head + 五维实例 embedding
  -> ISA：前景候选 + embedding FPS，最多 300 个实例 query
  -> 六层 256 维森林 query decoder
       第 1 层保留原 cross-attention
       后 5 层接 P3Former-inspired MFA 区域汇总/动态门控更新
       各层保留 self-attention、FFN、mask 与质量输出
  -> 整树 mask、语义 query、质量评分
  -> 整林提案合并、点归属、评价
```

主干、B、C、查询与掩码分支全部联合训练，没有冻结主干。坐标离散化、FPS 和最终 NMS 本身是离散操作，这与可训练端到端实例网络的常规含义一致。

ForestFormer3D 的完整 U-Net 与对应 decoder 源码已经随项目提交，不需要在服务器另克隆整个 OpenMMLab 工程。当前实现使用原生 torch/spconv 接入数据和监督。具体来源、固定上游 revision、修改与许可证见 [vendor 说明](../../tree_learn/forest_abc/vendor/README.md)。这是 ForestFormer3D 组件基础上的森林二分类组合模型，不宣称逐项复现官方训练配方。

相比官方工程，适配点包括：wood/leaf/ground 改成现有树/非树标签；torch FPS 代替新 FPS 扩展；显式近邻索引接 GeoMixer；独立实现 MFA 更新机制并加入空区域回退/残差混合；一对多关联直接使用查询来源体素的训练实例 ID；未知标签不参与损失；质量监督使用完整训练树点数修正裁剪影响。查询热身只发生在训练的前 2000 步，逐步从 GT 前景过渡到预测前景；验证和推理从不读取 GT 选 query。

## 输入与依赖

继续使用已有两个缓存目录中的 **coords.npy 和 labels.npy**：

- 源域：`data/candidate_mask_mlp_a1`，13 个训练森林、5 个验证森林。
- Wytham：`data/candidate_mask_mlp_a1_wytham_s42`。

保留这些缓存只是为了复用已对齐的坐标和标签，不读取其中的 `features.npy`、`votes_xy.npy`、`tree_probability.npy` 训练或预测。`baseline.npy` 只在预测完成后的指标对照中读取，不是模型输入。`cache.json` 和 manifest 仅用于定位与校验坐标来源。

标签约定：0 是非树，正整数是树实例 ID，负数忽略。输入体素的坐标代表点与抽样不依赖 GT，训练标签采用体素内已知标签多数；大块的训练体素随机抽样，并按纳入概率保留点数权重。整林输出仍回到原 0.1 m 点集，按已验证的相同对齐协议评价。

环境继续用服务器现有 `TreeLearn`，需要 CUDA torch、spconv、numpy、scipy、PyYAML；不需要新增 MinkowskiEngine、MMCV、torch-cluster 或专用边界算子。spconv 二进制与 GPU 是否兼容由服务器 `doctor/check` 确认，本地未执行确认。

默认每块训练最多 65536 个体素、推理最多 98304 个体素，保留 16 m 柱状范围。超限推理块保留空间代表后把全部原始点映射回预测体素，不直接截掉整林尾部；`prediction.json` 会报告抽样块数、最大输入体素数。GeoMixer 的邻居数为 16。运行串行读取块，不预先复制整林重叠 tiles，不开启大量 DataLoader worker。

## 服务器第一轮命令

先拉取本次分支，然后一条前台命令启动完整流程：

```bash
cd ~/projects/zrx/code/TreeLearn
conda activate TreeLearn
git pull --ff-only origin codex/candidate-mask-mlp-step1

bash tools/forest_abc/run.sh all
```

该命令始终在当前终端前台运行，逐阶段同时打印、保存日志：

1. `doctor`：显示 Python、torch/CUDA、GPU 和可用显存。
2. `prepare`：建立独立坐标索引，复用已有坐标缓存，无需重跑 TreeLearn 特征导出。
3. `check`：4096 体素小块、3 次训练更新，采用正式训练的 LR 热身，检查有限损失，并在三步窗口累计验证 **mask loss** 对主干、GeoMixer、MFA 的非零有限梯度证据。逐层检查连接与有限性，记录每层 MFA 的空区域数量；随后用无标签输入做一次前向。它不生成或选择正式模型权重。
4. `train`：从头联合训练完整组合 40000 步，前 1000 步 LR 热身；每 10 步打印 loss、query 数、学习率、耗时、内存/显存。
5. 每 5000 步进行完整 5 森林源域验证，按宏平均 F1 保存 `best.pth`。每 500 步保存断点，并在验证前先保存，以便验证中断后续跑。
6. `evaluate`：用源域选出的 best 对 5 个源域验证森林完整评估。
7. `external`：用同一个 best 对 Wytham 完整推理与评价，不使用 Wytham 选权重。

这是单套完整组合的训练与评价，不会自动启动 A、A+B、A+C 等消融。虽然 B/C 保留配置开关，默认均为 true。

默认从随机初始化训练，没有自动加载旧的冻结查询模型、TreeLearn 权重或官方 ForestFormer3D 预训练权重。因此不能保证很少训练步就超过已训练的 TreeLearn 基线。流程成功与模型有效分开判断：整林 F1、TP/FN 和 Coverage 才能说明实际变化。

## 中断后与单独运行

已有 `last.pth` 时继续完整流程：

```bash
bash tools/forest_abc/run.sh resume
```

恢复模型、AdamW 状态、步数与采样/torch 随机状态；若中断发生在验证阶段，会先补完该次验证，不重复已完成的训练步骤。配置或坐标/训练标签改变时会拒绝误续训，应使用新目录。

仅重新评估 best，不重新训练：

```bash
bash tools/forest_abc/run.sh evaluate
bash tools/forest_abc/run.sh external
```

可选模块单元测试，仍在服务器运行：

```bash
bash tools/forest_abc/run.sh tests
```

全套流程不要求该测试步骤先行；默认 `check` 专注于验证真实 CUDA 主干和梯度能否跑通。若准备阶段还没有 Wytham 缓存，`prepare` 会显示提示；应先恢复原有外部坐标缓存，再运行 prepare/external。

## 结果与日志

```text
logs/forest_abc_v1/latest/console.log       全流程日志
logs/forest_abc_v1/latest/train.log         本次训练实时输出
logs/forest_abc_v1/latest/status.txt        当前阶段与最终退出状态
work_dirs/forest_abc_v1_s42/check/report.json
work_dirs/forest_abc_v1_s42/training.jsonl
work_dirs/forest_abc_v1_s42/last.pth
work_dirs/forest_abc_v1_s42/best.pth
work_dirs/forest_abc_v1_s42/train_summary.json
work_dirs/forest_abc_v1_s42/evaluation/brief.json
work_dirs/forest_abc_v1_s42/external_evaluation/brief.json
```

首轮跑完主要贴 `train_summary.json`、`evaluation/brief.json`、`external_evaluation/brief.json`。如运行报错，保留终端 traceback 及对应阶段日志。

推理使用质量×掩码置信度排序、局部/全局 NMS，再进行点归属。分配后剩余不足原掩码一半、得分不足或点数太少的残片在占点前被拒绝，避免拒绝后留下不能再分配的洞；这些规则固定写在同一配置中。这是完整系统的一部分，首轮不单独调阈值来替代架构验证。

评价沿用 `aligned_voxels_ignore_negative_gt_iou_gt_0.5`，包括原有 FP 计数规则和按 GT 等权的 Coverage。Wytham 已参与此前开发分析，本轮仍按已使用的外部森林报告。

提交前仅在开发机完成了 Python 文件的 AST/编译静态检查、Bash `-n` 语法检查、文档相对链接和 Git 空白检查。没有导入模型执行，也没有在开发机运行上述单元测试或 CUDA 检查。

## 首次 doctor 的设备编号兼容修复

首次服务器运行停在 `torch.cuda.mem_get_info(torch.device('cuda'))`：该服务器版本要求设备带明确编号。现在在统一设备入口把 `cuda` 解析为当前逻辑设备（如 `cuda:0`），保留显式的 `cuda:1` 等设置，并同步设置当前设备；环境检查向设备名称/显存查询传整数编号并打印解析结果。这也覆盖后续训练、检查与评估入口，配置不需要修改。

该次失败尚未进入 prepare/check/train，不存在需要恢复的本次训练断点。拉取修复后仍运行 `bash tools/forest_abc/run.sh all`，不要使用 resume。补充了设备解析的 mock 回归测试，留在服务器执行，本地只做静态检查。

## check 反向传播的 Float/Double 兼容修复

第二次运行已通过 doctor 和全部坐标索引准备，停在 check 的 `loss.backward()`，报 `Found dtype Double but expected Float`。根据数据和损失路径，采样权重原表达式 `float32_array * total_voxels / selected_count` 在 NumPy 1.x 遇到较大的整数体素数时可能提升为 float64，随后把质量 IoU 目标带成 Double。相关标量提升背景见 [NumPy NEP 50](https://numpy.org/neps/nep-0050-scalar-promotion.html)。

修复包括：权重使用 float32 比例原位缩放；`tensor_block` 显式指定浮点输入/监督为 torch.float32，索引保持对应整数类型；损失内部将权重、完整树大小对齐预测类型，MSE 质量目标明确匹配 score 类型。check 首步打印 `CHECK_DTYPES`，包括输入、mask、score、总损失和各损失分量类型。

补充了大于 65535 体素的真实数据块、张量类型边界、Double 监督下完整损失反传三类回归用例，尚未在本地执行。已有索引可复用，配置和工作目录不用改；仍运行 `bash tools/forest_abc/run.sh all`，prepare 会复用完成的索引，然后重新 check。正式训练尚未开始，不使用 resume。

## check 的条件分支梯度判定修复

第三次服务器运行已完成首个真实训练步，日志显示 mask loss 对 backbone、geometry、mfa 的梯度 L1 分别约为 78.10、18.65、313.75，说明这条端到端路径至少在首块上已经接通。第二块却触发统一的 `Mask loss does not reach mfa`。旧检查把 None、NaN/Inf、精确零梯度都合并成这条报错，因此仅凭该日志无法确认第二块属于哪一种。

MFA 对空 hard-mask 区域会返回原 query：这种分支允许当前块的更新参数梯度为零，不能直接等同于计算图断开。现在每步仍立即拒绝 None 和非有限梯度，检查全部五层 MFA，而不再只抽第一层；对非零梯度按 backbone/geometry/mfa 三组在三步窗口累计取证。若某组整个窗口都没有非零证据，依然停止。逐层一直为零的探针保留在报告中，不用其他层的梯度掩盖 None 断路。

check 输出新增 `mask_gradient_by_probe_l1`、`zero_gradient_groups` 和 `mfa_regions`。`mask_gradient_l1.mfa` 改为五层探针的 L1 总和；`check/report.json` 保存 `gradient_audit` 累计结果。零梯度后仍会执行真实总损失反向传播和优化器更新；全参数非有限梯度检查保留。

同时修复 check 直接用 0.001 基础 LR、未走热身的问题：现在与正式训练相同，前三步 LR 为 0.00001，AdamW weight decay 也一致。没有改 MFA 的空区域回退、掩码阈值、正式模型结构或正式训练学习率。补充空区域零梯度、跨步证据、逐层断路及 NaN/Inf 回归用例，本地仍只做静态检查。拉取后重新运行 `all`，已有索引自动复用。
