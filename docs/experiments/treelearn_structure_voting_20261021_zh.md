# 冻结官方 TreeLearn 的结构连接与 XY 投票修正

## 本轮目标

首轮交付完整可训练、可验证、可进行 Wytham 推理的组合。沿用当前对齐点、忽略负 GT、IoU 严格大于 0.5 的评价协议，直接比较同一官方权重的缓存结果。先验证整个新增分支是否有效，再决定是否拆开消融。不重新训练已经失败的普通微调对照。

上轮 Wytham 官方到 combined 丢失 50 棵、恢复 9 棵，其中 47 棵丢失树的 z 跨度低于 10 m；41 棵属于大量未分配或完全未分配。结合已核查的导出路径，主要新增损失来自语义前景退化。本轮不再更新官方共享特征或语义路径。

## 实际架构

1. 官方输入仍为原 0.1 m 体素、常数特征，使用 `model_weights_with_small_20241213.pth`，校验 MD5。官方 input_conv、U-Net、output_layer、语义头、原三维 offset 头的参数和 BN 缓冲全部冻结。默认 FP32，与生产 pipeline 的官方前向一致。
2. 新分支读取停止梯度的官方末端 32 维特征、坐标、原投票 XY、树概率。图中没有 GT、树 ID、RGB、验证标签输入。
3. 在官方树概率不低于 0.5 的点中，每个 0.4 m 占用网格取一个实际点作为 support。support 数量没有全场截断；每个 query 取半径 1.2 m 内最多 8 个 support，排除自己，隔离 batch。
4. 连接 MLP 预测 query 与 support 属于同树的概率，以此加权邻居特征差和原投票位置差。原点保留隐式权重 1，无邻居时不发生除零。
5. 修正 MLP 输出 `1.5*tanh(raw)` 的 XY 残差，即每个坐标分量限制在 ±1.5 m；不是总向量长度限制为 1.5 m。最后一层权重和偏置为零，初始预测精确复现官方模型。
6. 修正后的 XY 接回原 offset 输出并走原聚类及点分配。官方语义 logits 和 offset Z 直接保留，原前景、verticality、Z 阈值及 seed eligibility 不变。聚类结果允许变化，这正是要检验的部分。

新分支约 5.7 千参数，准确数量由服务器 `check.json` 给出。训练只对采样 query 建立梯度；推理按 8192 query 分块，边特征不跨块累积，不建立全点 NxN 矩阵。GT 不参与 query 的邻域搜索或前向特征构造。

这里的“结构”是可学习局部归属和同树跨高度投票一致性；没有声称获得真实木质骨架、树干标签或复现 SATree。不会硬性要求树竖直、单干、每条枝干完整可见。

## 训练监督

保持上一轮 13 个训练森林、5 个验证森林，单独工作目录 `work_dirs/treelearn_structure_voting_v1_s42`。数据只复用原坐标缓存与官方 tiles，训练 crop 和 legacy 基部目标来自现有已使用的采样器。

- 每 crop 最多 64 棵树，每树最多 48 个 query。每棵树按 z 排序、等点数分层，每层抽 1 点，覆盖低、中、高位置；这是垂直跨度分层，不是地形归一化树高。
- 每树先平均、再在树之间平均，避免大树点数主导损失；连接目标也做同树/异树类别平衡与实例均衡。
- 连接正例为同 batch、同正实例 ID；已知其他树或背景为负例；负 GT 忽略。query 必须是内区有 offset 监督的树点。跨 batch 不构造连接。
- `L = L_XY + 0.3 L_connection + 0.1 L_vote_consistency + 0.02 L_residual`。
- XY 为修正偏移对 legacy 基部偏移的 Smooth L1；一致性为同树采样点的预测投票位置接近其均值；残差正则限制对官方预测的改动。没有要求各点的 offset 向量相同。
- AdamW，新增分支学习率 3e-4，30 步 warmup，余弦下降，梯度范数裁剪 1，共 3000 步；没有后续解冻阶段。
- 500 步只评 G4N，若 F1 比官方低超过 5 个百分点则停止。1500、3000 步评完整 5 森林，以验证 macro F1 选择 `best.pth`。Wytham 不参与选择。

本轮保留官方前景，因此不能恢复原官方模型根本没有保留足够树点的实例。当前训练数据小树覆盖不足也没有被结构分支自动解决。冻结前景是防止上轮退化，不是保证 F1 不下降；错误的 XY 修正仍可能改变聚类而掉点或合并。

## 服务器前台执行

在 Linux 服务器操作。不要同时启动其他 GPU 实验，启动脚本与已有 ForestABC 共享 `flock` 锁。

```bash
cd ~/projects/zrx/code/TreeLearn
conda activate TreeLearn
git pull --ff-only origin codex/candidate-mask-mlp-step1
bash tools/structure_voting/run.sh all
```

`all` 顺序为 doctor → 新增 tensor/graph 单测 → prepare → GPU check → train。全过程在当前终端显示，同时保留各阶段日志和 `/usr/bin/time` 记录。不会后台启动。

GPU check 先用 G4N 最大生产 tile，以及外部缓存存在时的 Wytham 最大 tile，检查初始语义和完整偏移精确等于官方。再执行 4 个真实训练 crop 更新，检查官方参数/BN 全部不变、语义/Z 不变、XY 确实更新、仅投票损失能通过连接 MLP 反传，并检查权重与优化器加载。

中断后保持代码、配置与数据不变，前台继续：

```bash
bash tools/structure_voting/run.sh resume
```

训练成功后，读取两个全量验证结果：

```bash
cat work_dirs/treelearn_structure_voting_v1_s42/validation/step_001500/brief.json
cat work_dirs/treelearn_structure_voting_v1_s42/validation/step_003000/brief.json
cat work_dirs/treelearn_structure_voting_v1_s42/train_summary.json
```

需要测试 Wytham 时，运行源域验证选出的 `best.pth`：

```bash
bash tools/structure_voting/run.sh external
cat work_dirs/treelearn_structure_voting_v1_s42/external_evaluation/brief.json
cat work_dirs/treelearn_structure_voting_v1_s42/external_evaluation/wytham_vox0.1/foreground_identity.json
```

如需重新独立评估 `best.pth` 的 5 个验证森林：

```bash
bash tools/structure_voting/run.sh evaluate
```

状态文件：`logs/treelearn_structure_voting/latest/status.txt`；完整实时输出：同目录 `console.log`，训练阶段 `train.log`。Wytham 阶段为 `external.log`。`DONE` 仅表示流程完成，效果以指标为准。

## 怎么判断

先看 `foreground_identity.json`：`changed_foreground_points` 必须为 0。评价逐块比较最终预测是否为正实例与官方缓存的对应布尔值；出现差异直接停止，先核对 pipeline，不能用改动后的前景冒充冻结对照。

再看 F1、completeness、commission、Coverage，辅助看 point precision/recall 和 TP/FP/FN。希望在前景不变的前提下，投票分配更正确。F1 单独增加而 Coverage、completeness 明显下降，需要说明取舍，不视为全面改善。

训练日志给出树均衡的 `base_vote_loss` 和 `vote_loss`、连接正负边数量、连接准确率、平均 XY 残差。训练投票误差下降并不等于全林分割提升；连接准确率也不跨 batch 或把 unknown 当负例充数。

目前尚无这一版模型的实测结果，不预先给出增益承诺。Wytham 已被用于多次开发诊断，结果应称为外部数据集开发评估，不称为完全未接触的盲测。5 森林验证仍沿用当前官方权重参照及其数据重叠限制，不直接对标论文 Table 3 的独立测试数值。

## 本地验证范围

本地只进行源码静态审查、Python 3.10 语法检查、配置引用检查、Bash 语法检查和 Git 空白检查。不在本地执行单元测试、模型前向、训练或推理。新增测试与完整 GPU check 由上面的服务器命令执行。
