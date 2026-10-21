# TreeLearn 小改：多尺度残差融合与实例辅助监督

日期：2026-10-21。状态：首版服务器 11 项测试和真实切块 check 通过后，control 曾因非有限梯度退出；稳定性修复后已收到两组 3000 步训练和 Wytham 外测结果，见下文。新增 CPU 掩码诊断待服务器执行。本地没有运行模型、训练、推理或单元测试。

本轮直接实现完整小组合，并配相同步数的普通微调对照。不使用 ForestABC/V2b 权重。两组均从官方 `model_weights_with_small_20241213.pth` 初始化，MD5 固定为 `38266e515dc387ac5ca32619e791cc20`，旧模型张量必须全部按键名和形状加载成功。

## 结构

- 原版：32 通道、七级 Sparse U-Net、0.1 m 体素、原语义头和三维 base-offset 头。
- 新增分支读取 decoder level 1/2 的 64/96 维特征，各投影到 16 维，按 `[batch, spatial coordinates]` 查找 stride-2 卷积父体素并对齐最终精细特征。不能按特征行号拼接。
- 拼接最终 32 维和两个 16 维分支，经 64→64→32 的小 MLP 得到 ΔF；`F′ = F + tanh(γ)ΔF`，γ 初始为零。新增参数约 1.06 万，实际数量由服务器 check 报告。
- 原语义头和 offset 头共同读取 F′。一个 32→32→8 的实例辅助头也读取 F′，仅在训练时计算。
- 保留原始语义损失（乘 50）及 L2 offset 损失；增加 `0.05 × L_instance`。L_instance 包含同树特征聚集、异树中心分离和空间相近的异树点分离。实例/近邻采样按 batch 隔离，未知标签和背景不充当异树负样本。
- 每个 crop 最多取 64 棵树、每树 48 个按高度分层采样的点，最多 2048 个边界负样本对；不构建完整点云 N×N 距离矩阵。该采样上限只用于辅助损失，不截断模型输出或评价树木。
- 推理不计算实例辅助头，继续使用原 TreeLearn 语义判定、base seeds、HDBSCAN、剩余点分配和切块平均。

参考：[ForestFormer3D 作者代码](https://github.com/SmartForest-no/ForestFormer3D) 的实例表征思路；[CBL 作者代码](https://github.com/LiyaoTang/contrastBoundary) 的边界对比思路。这里的残差融合与实例边界采样是本项目适配实现，不声称原样复现 CBL，也不声称已验证可提升森林指标。

## 两组训练

| 项目 | control | combined |
|---|---|---|
| 起始权重 | 同一官方权重 | 同一官方权重 |
| 前 300 步 | 普通语义/offset heads 微调 | 原模型冻结，仅新增分支与辅助头学习 |
| 后 2700 步 | heads＋最后两级 decoder 微调 | 相同原模块＋新增模块联合微调 |
| BN running statistics | 全部冻结 | 全部冻结 |
| 数据、总更新步数 | seed 42，同一 3000 个 crop 计划 | 完全相同，可核对 crop_sha256 |
| 辅助实例监督 | 无 | 有 |

两组的总更新预算相同，第一阶段的训练参数不同。比较报告同时提供 1500/3000 相同步数差值，以及两组各自用相同源域候选步数选择的最佳结果，不能把差值直接全部归功于某一个新增模块。

训练从现有 `data/candidate_mask_mlp_a1` 的坐标和训练标签在线取 crop；不读冻结模型特征、votes 或候选预测作为训练输入。沿用 13 个训练森林和 5 个验证森林。上下文方形边长 35 m、中心监督边长 8 m，温和旋转/镜像及 0.95–1.05 缩放，每块最多均匀保留 393216 点。legacy base 目标在完整 crop 上计算后才采样，两组完全一致。这是本轮共同微调配方，不声称逐项复现官方原始训练。

第 500 步各跑一次 G4N 完整 pilot；如果 F1 比官方缓存低超过 15 个百分点，停止该组并保存失败信息。此阈值只拦明显损坏，不代表通过就有效。1500/3000 步分别跑全部五个验证森林，仅据五森林 macro F1 选择 best.pth。

## 首次 control 非有限梯度的修复

首次在普通微调 control 训练期间退出，combined 尚未开始正式训练，因此该错误不是新增分支的性能失败。现有终端日志没有记录坏梯度参数及当时的 AMP 缩放值，无法确认直接触发原因。

- 原实现遇到非有限梯度直接抛错，阻止了 GradScaler 正常跳过坏更新并降低缩放。现改为同一 crop 最多额外重试两次 FP16，然后以未缩放 FP32 复核。坏尝试不裁剪、不更新参数、不推进步数和学习率；Torch RNG 恢复，两个 arm 仍使用同一 crop 计划和成功更新预算。
- 原 L2 offset 写法 `sqrt(sum(error**2))` 在误差恰好为零时存在反向 `0 * inf` 隐患。改为 `torch.linalg.vector_norm(error)`，保持欧氏距离目标，在零点使用有限的零次梯度。两组共同使用该修复；推理路径不受影响。
- 训练日志增加 `loss_scale`、`update_precision`、`update_attempts`、`amp_retries` 和 `fp32_fallback`。`numerical_events.jsonl` 记录重试、出错参数名、NaN/Inf 数量、crop 标识及最终恢复；FP32 仍异常则明确失败，不用填零或跳过样本掩盖问题。
- 新回归测试覆盖零/极小 offset、AMP 坏更新拒绝、原 crop 重试、FP32 回退、FP32 持续失败及真实 CUDA GradScaler；扩展生产 check 为原有 combined 检查加 32 次 control warmup 更新与 2 次 decoder 联合更新。该检查只能验证覆盖到的情况，不能保证任意训练步都不会再出现数值问题。

本次使用新目录 `work_dirs/treelearn_residual_instance_stable_s42`，保留失败的 `work_dirs/treelearn_residual_instance_s42`。代码及损失实现已变化，请拉取后执行 `all` 重新 check/初始化，不对旧目录执行 resume。坐标索引和原版 tiles 继续复用。

AMP 行为依据 [PyTorch 2.0 GradScaler 源码](https://github.com/pytorch/pytorch/blob/v2.0.0/torch/cuda/amp/grad_scaler.py)；遇到 Inf/NaN 时原生行为是跳过该次优化器更新并降低缩放。

## 服务器执行

先停止此前实验，保持当前终端前台运行：

```bash
cd ~/projects/zrx/code/TreeLearn
conda activate TreeLearn
git pull --ff-only origin codex/candidate-mask-mlp-step1
bash tools/treelearn_residual_instance/run.sh all
```

all 顺序执行 doctor、单元测试、坐标索引准备、真实生产切块检查、control 训练与验证、combined 训练与验证、配对汇总。没有后台进程；所有子流水线输出也会显示在终端。

使用现有 `resolved_configs/<plot>.yaml` 指向的官方 tiles，不重新生成切块。若这些文件被清理，doctor 会明确报告缺失，不能换成 ABC 的粗采样推理。在自己的输出目录创建只读用途的 tiles 软链接，不改写已有缓存。

check 在 G4N 和已有 Wytham 最密集的真实切块上，分别检查 FP32 和混合精度的初始预测与官方模型逐值相同，以及父体素索引完整；另检查 34 次 control 更新、两个分支训练更新、联合阶段梯度、FP32 辅助损失单独回传至 decoder，以及模型/优化器保存载入。check 权重不用于正式训练。正式训练仍重新读取官方权重。

查看结果：

```bash
cat work_dirs/treelearn_residual_instance_stable_s42/comparison.json
cat work_dirs/treelearn_residual_instance_stable_s42/control/train_summary.json
cat work_dirs/treelearn_residual_instance_stable_s42/combined/train_summary.json
```

关键字段 `combined_minus_control_pp` 比较小组合与普通微调；各组 `delta_pp` 比较官方缓存。必须同时看 F1、Completeness、Commission、Coverage 和点级 Precision/Recall，不能只看减少了多少预测树。

中断后续跑：

```bash
bash tools/treelearn_residual_instance/run.sh resume
```

resume 会恢复参数、优化器、混合精度 scaler 和 RNG；后续 crop 计划由 seed/步数独立确定。固定验证快照不覆盖，已完成森林评价可复用。数据、配置、官方权重或执行代码变化时拒绝静默续跑。

源域结果完成后，可分别用两组源域选出的 best.pth 做 Wytham 对照：

```bash
bash tools/treelearn_residual_instance/run.sh external
cat work_dirs/treelearn_residual_instance_stable_s42/control/external_evaluation/brief.json
cat work_dirs/treelearn_residual_instance_stable_s42/combined/external_evaluation/brief.json
cat work_dirs/treelearn_residual_instance_stable_s42/external_comparison.json
```

新预测必须与原缓存坐标、点顺序及 GT 标签逐块完全一致，否则不计算差值。延用当前 `aligned_voxels_ignore_negative_gt_iou_gt_0.5` 口径，negative GT 忽略、IoU 严格大于 0.5、五森林等权平均。新输出只保存必要预测数组，不再导出 32 维整林特征，降低内存与磁盘占用。

Wytham 已参与本项目开发分析，不能把本轮称作全新盲测；官方预训练与各评价森林的重叠问题仍沿用已有数据审计说明。训练完成和 EXIT=0 仅表示流程完成，有效性以配对结果为准。

## 首轮完成结果与最终掩码诊断

服务器回传：两组源域最佳 checkpoint 都为 3000 步。五森林 macro F1 为官方 88.1946%、control 88.6739%、combined 88.6237%。组合相对 control 为 −0.0501 pp F1、−0.2690 pp Coverage、−0.5802 pp 点召回；该单种子结果没有新增组合收益的证据。

Wytham 同口径结果：

| 指标 | 官方 | control | combined |
|---|---:|---:|---:|
| TP / FP / FN | 596 / 222 / 281 | 555 / 175 / 322 | 555 / 173 / 322 |
| F1 (%) | 70.3245 | 69.0728 | 69.1589 |
| Completeness (%) | 67.9590 | 63.2839 | 63.2839 |
| Commission (%) | 27.1394 | 23.9726 | 23.7637 |
| Coverage (%) | 65.1253 | 61.1164 | 60.8451 |
| 点精度 (%) | 71.2304 | 65.4641 | 65.3757 |
| 点召回 (%) | 80.3670 | 73.6913 | 72.5107 |

共同微调流程已经出现迁移退步，新增组合相对对照只减少 2 个 FP，TP 总数相同，点召回还低 1.1805 pp。两个 TP 数量相同并不能证明检出的是同一批 GT；官方 TP 到两组 TP 的净下降 41 也不能直接写成只丢失 41 棵，可能同时有更多丢失和部分恢复。当前没有依据把退步直接归因于小树、合并、遗忘或某个单独模块；结束当前组合训练，先诊断已有预测。

只读取已有最终掩码，前台运行 CPU 诊断，不重新推理，也不加载权重或缓存网络特征：

```bash
cd ~/projects/zrx/code/TreeLearn
conda activate TreeLearn
git pull --ff-only origin codex/candidate-mask-mlp-step1
(
set -euo pipefail
python -m unittest discover -s tests -p 'test_residual_instance_diagnostics.py' -v
mkdir -p logs/treelearn_residual_instance
python -u tools/treelearn_residual_instance/diagnose.py \
  2>&1 | tee logs/treelearn_residual_instance/external_diagnostics.log
cat work_dirs/treelearn_residual_instance_stable_s42/external_diagnostics/wytham_vox0.1/brief.json
)
```

工具复用已回归的 ForestABC 有界计数与匹配核心，兼容这次的 `predictions.npy` 和 TreeLearn 轻量导出元数据。三个模型只扫描一次共同计数，逐一复现已保存指标后才输出诊断。配对包含官方→control、官方→combined、control→combined，输出 lost/recovered、丢失树最终错误类型、z 跨度分组和 GT 树点 assigned/unassigned 转换；另统计最佳 IoU 在 0.5 附近的数量。实例编号不跨模型直接比较。

逐树详情在 `*_trees.csv`，完整分组指标和输入记录在 `summary.json`。`gt_point_assignment_fraction` 是真实树点被分配到任意正实例的比例，不代表归属正确。高召回低纯度也不能单独证明两树合并；最终未分配点也不能定位语义头、seed、聚类或剩余点分配的具体原因。该工具不改动模型、配置或训练运行代码，不使已完成 checkpoint 的代码来源记录失效。

## 本地交付验证范围

仅进行 Python 3.10 语法/导入路径静态检查、Bash 语法检查和 git diff 空白检查。本次新增 4 项 CPU 诊断回归尚未在服务器执行，原模型测试与 GPU check 不由该诊断重新触发。模型修改为默认关闭的可选路径，原版未启用时不创建新增参数。
