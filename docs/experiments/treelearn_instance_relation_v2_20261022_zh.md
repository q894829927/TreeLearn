# 官方冻结点特征与局部连接关系头 V2

## 范围与 V1 的差别

官方 U-Net 完全冻结，不加载或重新执行官方网络，输入使用已经导出的官方预测、XY 投票、语义概率和 32 维点特征。V1 的候选节点、候选边、4 m 几何规则和不连锁的局部 union writer 保持不变；新增头只控制接受哪些原实例对。未选原实例的编号和完整点集逐点不变，官方前景逐点不变。

V1 的六组关注碎片都已入图，但在分数门槛前被筛光；12 个选中原 TP 中 11 个实际超过保护标准却被质量头低估。V2 因而改点级关系编码、训练样本与动作监督，不再把 V1 的原实例 IoU、union precision/recall/IoU/gain 回归值分别作为部署硬门槛。

这版是官方 TreeLearn 预测后的可学习局部修复。尚无服务器实测增益，不能承诺恢复六组案例或 Wytham 不丢树。

## 预测输入与局部结构

复用 `data/treelearn_instance_relation_v1_s42/` 的 GT 无关几何与候选缓存，按原全局索引 hash 重建每个实例每高度层最多 64 个点的索引。校验重建坐标、完整人口统计与 V1 一致，再从官方 `features.npy` 用 mmap 读取这些索引对应的 32 维特征。候选不按 GT 增删，不重跑全场预测。V1 的 `best.pth` 不参与 V2 初始化或部署。

每实例默认最多 256 个点。原始每点 38 个通道为 XYZ、XY 投票、树概率、冻结特征32维；网络转换为相对坐标、投票偏移、实例自身和 pair 共享的相对高度，以及标准化的冻结特征，共42维。归一化均值、尺度只来自训练域实际点样本并写入 checkpoint。

点 MLP 编码后，双向建立4个几何近邻和2个投票近邻，去掉重复点对。连接编码融合两端点特征和相对距离；分别汇总全局接触、≤1 m 接触区、相邻高度的近 XY 连接、两端下部连接，并保留每节点下部支撑特征。下部只是自身可见高度的低四分之一，不冒充真实树干标签，也不硬编码“双峰一定是两树”。

该编码在实例交换时对称，坐标与投票一起平移时保持相对几何。无需 54/59 维统计摘要作为分类输入；旧统计只服务于已保留的候选与训练负例排序。默认可训练参数静态公式为119,561，准确数量由服务器打印。

部署前向只接收 `points / valid / weights`；不接收 GT、实际 IoU 或原 TP 标记。未知标注点可作为输入上下文，但不作为连接监督的负例。

## 真实动作标签与合成训练数据

源域原候选的有效训练 pair，逐一使用当前最大总 IoU Hungarian 匹配协议重放单次 union，计算实际找回、丢失、TP/FP变化。原本某个 union IoU >0.5 并不直接等同于检测恢复。真实对的动作分为：

- keep：不符合有益同树合并条件，或导致丢树；
- recover：满足有益同树条件，实际完整匹配找回至少一棵树且不丢原树；
- complete：满足有益同树条件且不丢树，但没有新增检出；包括已有实例补全和有益碎片清理。

误并风险监督为实际丢树或不符合同树标签。动作价值为 `clip((找回数 − 2×丢失数 − 0.25×FP增量)/2, −1, 1)`。union precision/recall/IoU 只作辅助回归监督。

仅在13个训练森林建立带GT的局部点库，使用已经被官方前景覆盖的 GT 点样本；实际全量 GT 前景覆盖率至少0.7且样本至少48点才进入库。每树最多512点。验证与 Wytham 不建立该库，不生成合成候选。

正例由水平或随机 XY 平面切断、空间遮挡、密度抽稀生成，也混入少量实际邻树点。空间遮挡会减少代表质量，密度抽稀保留所代表的人口质量。增强后重新计算加权虚拟掩码的匹配与动作标签，不满足有益条件的样本改标 keep，其他样本也按实际算出的类别标注，不强行保留原期望标签。合成标签是加权局部观察的虚拟目标，不是全林实测指标；始终与真实错误对混合使用。

困难负例包括源域候选中距离近且两个端点原本均为 TP 的组合，以及邻近 GT 树的完整点集。每1000步从训练域真实负例中挖掘当前高分误判对。GT只用于训练抽样、标签与评价，不进入部署输入。

## 模型、损失和接受规则

默认64维点/连接编码、约12万参数，FP32训练。输出保持/恢复/补全三类动作分数、误并风险、动作价值、辅助union质量，以及点对连接监督的logits。

损失为动作交叉熵 + 风险BCE + 2×价值SmoothL1 + 0.25×辅助质量SmoothL1 + 0.5×点连接BCE。点连接只在两端均有已知标注时监督，GT<0排除，GT=0为已知背景。连接编码同时进入动作头，辅助梯度能够影响最终预测。

每批24对，真实对和合成对各半。真实恢复/补全分开抽样，负例优先用困难池；某一真实类别不存在时明确回退并打印实际类别计数。合成部分也打印实际增强后的类别计数。AdamW学习率3e-4，共4000步，每500步验证。

只用一个接受分数：`recover_score + 0.25×complete_score + 0.5×utility − 2×risk_score`。原始softmax/sigmoid分数不解释为真实正确率。阈值候选 `[0.25,0.5,0.75,1.0,1.25]` 和显式 keep-all 在五个源域验证森林的自然候选分布上选择；要求源域不丢树且macro F1不下降，然后依次优先找回数、F1、Coverage、较少编辑。另报自然候选中恢复类别的precision/recall、有标签接受精度、恢复pair召回、无标签接受数和两个原TP被合并次数。Wytham不参与选阈值。

通过阈值的候选沿用双方互选、一轮内每原实例最多参与一次的规则。写出复用 V1 writer，只把选中两片中较大 ID 映射到较小 ID。不重新聚类或重新分配全林点，也不拆分已经混合的实例。

## 服务器前台执行

需要已有官方32维缓存及 V1 完成的几何、监督缓存。`doctor`先检查这些输入，测试失败即停止。整个流程保持前台输出，并写各阶段日志与资源时间记录。本地仅做源码静态检查，没有执行模型或单元测试。

```bash
cd ~/projects/zrx/code/TreeLearn
conda activate TreeLearn
git pull --ff-only origin codex/candidate-mask-mlp-step1

bash tools/instance_relation_v2/run.sh all
```

`all` 顺序：doctor → 服务器回归测试 → 点样本/真实动作标签/训练点库准备 → 实际数据梯度和对称性检查 → 训练 → 源域验证 → Wytham预测与评价。一次性CPU标注会逐对重放匹配，打印 `COUNTERFACTUAL_TARGETS`；评分打印 `SCORE_PAIRS`，不是后台运行。无需重跑原 U-Net 的全林推理。

默认输出：

- 预处理：`data/treelearn_instance_relation_v2_s42/`；
- 模型与结果：`work_dirs/treelearn_instance_relation_v2_s42/`；
- 前台日志：`logs/treelearn_instance_relation_v2/latest/`。

准备阶段中断且尚无 `last.pth` 时重跑 all；已有训练 checkpoint 后继续：

```bash
bash tools/instance_relation_v2/run.sh resume
```

仅复评最佳模型或不带GT部署：

```bash
bash tools/instance_relation_v2/run.sh evaluate
bash tools/instance_relation_v2/run.sh external
bash tools/instance_relation_v2/run.sh predict
```

predict可以在没有外部 `labels.npy` 的情况下运行。external先保存预测，再读取已有GT监督评价。保持代码/配置和缓存不变以复用同一实验，修改来源时使用新目录；V1训练文件及其签名不改。

## 回传结果与判断

```bash
cat work_dirs/treelearn_instance_relation_v2_s42/train_summary.json
cat work_dirs/treelearn_instance_relation_v2_s42/evaluation/brief.json
cat work_dirs/treelearn_instance_relation_v2_s42/external_evaluation/brief.json
```

先看真实训练与验证的 recover 类是否存在，再看实际找回/丢失、接受合并数、误并两个原 TP 的数量以及 F1/覆盖率。`no_edit_result=true` 明确表示回退，不能称作提升。每森林另有 `accepted_merges.json`、`action_diagnostics.json`、`all_gt_trees.csv` 和 Wytham 六组 `watchlist.json`，便于核对预测风险与实际损益。

服务器13项测试覆盖候选与冻结通道对齐、GT隔离、真实反事实与完整匹配一致、未知点、合成标签/密度人口质量、实例交换与平移、点分支梯度、归一化保存、单分数策略、非空union的不变约束，以及小流程训练/挖掘/续训/无GT推理。本地没有执行这些测试；源码兼容性和 Bash 语法检查不等于服务器运行成功。
