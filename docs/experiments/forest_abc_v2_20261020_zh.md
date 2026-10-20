# ForestABC V2：中心约束、完整注意力和实例归属竞争

本版基于用户在服务器提供的 V1 20000 步结果设计。开发机只编辑代码并做静态检查，不执行模型、训练、推理或测试；CPU 回归和 GPU 检查都由下列服务器流程运行。效果尚未验证。

2026-10-20 首次服务器运行：doctor 通过，12 项回归有 11 项通过，中心目标测试因参考均值直接用 float32 累加而失败，差异约 1.43e-6 m。生产代码的 float64 分块累加没有改变；测试改为 float64 求参考均值、最终转 float32 后逐值精确比较，并检查保存 dtype。此次在 tests 阶段停止，尚未进入真实数据准备、GPU check 或正式训练。拉取修复后重新执行下文 `all`，由服务器复验。

## 诊断依据

V1 五样地验证 F1 为 92.11%，官方权重缓存基线为 88.19%；Wytham 却为 44.88%，低于基线 70.32%。最终掩码诊断显示 570 棵漏检中有 430 棵属于高召回、低纯度，实例混合是主要表象。

22 块、12 棵目标的阶段重放已由用户完成，已检查的分数/点数/bbox 没有元数据不一致。8 棵失败树分为：

| GT | 最终头最佳 IoU | V1 局部 NMS 后最佳 IoU | 保存整林最佳 IoU | 线索 |
| --- | ---: | ---: | ---: | --- |
| 10 | 0.0654 | 0.0653 | 0.0654 | 所抽块无合格最终头掩码 |
| 275 | 0.1839 | 0.0642 | 0.0642 | 初始头曾有 0.5467，第一层后降至 0.3844 |
| 628 | 0.1124 | 0.1095 | 0.0997 | 所抽块无合格最终头掩码 |
| 403 | 0.3244 | 0.3224 | 0.3168 | 所抽块无合格最终头掩码 |
| 517 | 0.2498 | 0.2498 | 0.3104 | 所抽块无合格最终头掩码 |
| 194 | 0.8199 | 0.7852 | 0.1900 | 局部候选合格，最终结果差 |
| 690 | 0.7052 | 0.5359 | 0.4254 | 同上，另有 2 条所抽高分掩码冲突记录 |
| 854 | 0.5661 | 0.5661 | 0.1539 | 同上，另有 1 条冲突记录 |

每棵失败树都有至少两个覆盖其 95% 点的重放块，有 2–8 个包含目标点的 query 输出细胞。不是完全没有观察到目标或完全没有 query 覆盖。细胞归属不等同于实际代表点标签。4 棵成功对照也出现抑制冲突；没有观察到已接受好候选因分配而退化的记录，因此不能只归罪于 NMS 或 MFA。这是错误富集抽样，不能把 5/8 和 3/8 外推为整林比例。

## 完整组合

代码为 `tree_learn/forest_abc_v2/`，运行入口为 `tools/forest_abc_v2/run.py`。V1 源码与配置不修改。

1. 保留现有 sparse U-Net、GeoMixer、语义与 embedding 头、ISA 选点。新增 XY 中心偏移头。训练目标是训练缓存中每棵完整标注树点的 XY 质心，不是树干位置，也不使用 Wytham/验证树中心。
2. decoder 六层均保留 cross-attention；后五层额外接入 MFA 残差。新增坐标位置编码和预测中心间距离的掩码偏置。初始头与六层头均受监督。各层含 self-attention/FFN，诊断变化不能当作模块消融结论。
3. 前六个辅助头继续锚点式一对多掩码监督；最终头做 Hungarian 一对一匹配，并让全部实例 query 与背景在有效点上进行归属 softmax。最终质量头学习互斥归属后的完整树 IoU，已知区域中的重复 query 质量目标为零；未匹配的未知标签 anchor 不作为质量负样本。负 GT 点不计损失或评估。
4. 树实例等权：有效点损失中各可见树分到相同正样本总权重，存在背景时背景占 25%；中心偏移也按树均衡。训练尺度 0.5–1.25、体素保留比例 0.4–1.0，保留已知点质量的采样修正。
5. 推理从最终头生成候选，按预测中心距离不超过 1.5 m、交集占两个掩码各至少 60% 判断重复。局部和跨块均融合观测，不因单向包含而直接删除小候选。固定分组代表避免距离链式合并。融合与中心距离都是启发式规则，仍可能错合或错分，需验证。
6. 每点证据为 `sigmoid(mask_logit - background_log_odds + 0.25*log(query_score))`。观测重叠取最大值，不按块数累加；不同组逐点竞争。第一次分配后按最小点数、保留率和组分数验收，再移除碎片重新竞争。验收条件对剩余组获得更多点是单调的，第二遍即可完成回收。

## 初始化、预算和结果口径

- 从 `work_dirs/forest_abc_v1_s42/best_step_020000.pth` 加载所有兼容旧张量：骨干、GeoMixer、点头、decoder 预测头、首层 cross-attention、MFA 等。新增后五层 cross-attention、位置编码和中心头独立初始化；MFA 残差强度重设为 sigmoid(-4)，避免在新增注意力路径上立刻施加强更新。全部参数共同训练。
- V2 的步数从 0 开始，默认再训练 20,000 步；这不等于总共只训练 20,000 步。学习率 0.0003，前 500 步 warmup。继承自 V1 的数据与训练成本须计入报告。
- 六层 cross-attention 的训练显存更大，启用 decoder 梯度检查点，将训练体素上限设为 49,152；推理仍为 98,304。服务器 `check` 会实际跑小块梯度、生产训练规模、序列化恢复后无 teacher 训练，以及源域和 Wytham 各一个密集块的推理。检查权重不会用于正式训练。
- 每块最多保留按分数排序的 96 个候选，裁掉数量记录为 `topk_pruned`，属于显式推理策略。实例成员在磁盘保存 int32 索引和 float32 点证据，最多 10 亿条、约 8 GB；分配另有三个 N 点数组，Wytham 约 447 MB，尚不含输入/模型。需要额外磁盘空间。预算溢出会明确失败，不截断成员。
- 共用 V1 GPU 流程锁。新工作目录 `work_dirs/forest_abc_v2_s42`、日志 `logs/forest_abc_v2`，复用坐标索引并仅为训练数据新增 `v2_centers.npz/json`，不改原坐标、标签、基线、预测或 V1 权重。
- 每 5,000 步验证五个源样地，按其 macro F1 选 best；同时保存该步固定的 `step_XXXXXX.pth`。不按 Wytham 选 checkpoint。Wytham 已参与设计诊断，后续结果应称开发集验证，不能称全程未见的最终测试；正式结论需要另一份未参与设计的外部测试集。
- 验证/外部评估统一沿用现有负 GT 忽略、严格 IoU > 0.5、低树点占比排除的协议；比较 F1、检出率、误检率、Coverage、点精确率与召回率。V2 是否同时改善实例检出与纯度，要由实际结果回答。

## 服务器执行：全程前台

确认当前 V1 训练已停止，冻结的 20000 步文件仍在。执行：

```bash
cd ~/projects/zrx/code/TreeLearn
conda activate TreeLearn
git pull --ff-only origin codex/candidate-mask-mlp-step1
bash tools/forest_abc_v2/run.sh all
```

顺序是 doctor → CPU 回归 → prepare（训练中心统计）→ GPU check → 正式训练 → best 五样地评估 → best Wytham 评估。终端会持续打印阶段、训练损失及预测/融合/两遍分配进度，任一步失败立即停止，完整输出也保存到日志。CPU 用例覆盖中心目标变换、未知点忽略、树均衡、唯一匹配、重复 query 竞争、保留小掩码、非链式融合、碎片回收、无 GT 的完整预测、chunk 一致性和预算失败。

如果需要分开检查后再训练，依次执行 `doctor`、`tests`、`prepare`、`check`、`train`、`evaluate`、`external`；替换上面命令最后的 `all` 即可。首次运行优先用 `all`。

训练中断后恢复：

```bash
cd ~/projects/zrx/code/TreeLearn
conda activate TreeLearn
bash tools/forest_abc_v2/run.sh resume
```

恢复核对配置、中心目标、数据、源码和推理策略，恢复 optimizer、采样 RNG、CPU/CUDA RNG，并补跑中断的验证。不要用 V1 的 `resume` 继续 V2。

查看阶段和首轮验证：

```bash
cat logs/forest_abc_v2/latest/status.txt
cat work_dirs/forest_abc_v2_s42/check/report.json
cat work_dirs/forest_abc_v2_s42/validation/step_005000/brief.json
```

完整流程完成后：

```bash
cat work_dirs/forest_abc_v2_s42/train_summary.json
cat work_dirs/forest_abc_v2_s42/evaluation/brief.json
cat work_dirs/forest_abc_v2_s42/external_evaluation/brief.json
```

`CHECK_PASSED` 仅表示检查流程成功，`DONE` 仅表示工作流完成，两者都不代表效果提升。先用完整组合看是否改善，再考虑拆模块。
