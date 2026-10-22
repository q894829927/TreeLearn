# 实例关系 V3：统一恢复监督、真实难例与分层结构

## 针对已经确认的问题

V2 在 Wytham 找回 1 棵、丢失 8 棵；8 个有害动作的预测风险分数为 0.004–0.045。
六组目标碎片均有候选，五组可独立恢复的动作却被打低分。
源域还存在实际恢复动作被旧纯度/增益条件标成 keep 的情况。
V3 集中修改这三点，保留原候选和局部合并写出机制。

## 1. 真实与合成样本采用同一套损益标签

| 类别 | 唯一的动作判据 | 部署动作 |
|---|---|---|
| other，编号 0 | 无丢失、无恢复 | 保持 |
| recover，编号 1 | 无丢失、至少恢复一个 GT | 可以申请合并 |
| destroy，编号 2 | 至少丢失一个原已检出 GT，即使同时恢复其他 GT | 保持 |

真实标签复用 V2 已保存的完整计数反事实：最大总 IoU Hungarian 匹配后按严格 IoU > 0.5 检测。
合成标签在加权合成观察的两实例场景中用相同的损益优先级重算，不宣称等价于真实全森林观察。
真实标签和合成标签都不再要求通过旧的同树纯度、union 纯度或最小 IoU 增益条件。
纯度、完整度及 union IoU 继续作为辅助质量目标；单纯减少 FP 不再得到恢复标签或独立收益奖励。

模型只独立预测一个三类动作分布及辅助质量。原日志中的 `risk` 和 `utility` 分别从三类分布导出，
不再训练独立的风险/收益回归头。
部署要求预测类别为 recover，且 `p(recover) - 3*p(destroy)` 达到源域选定阈值；
辅助质量不能否决或补偿这个判断。输出分数没有被宣称为校准后的实际错误概率。

## 2. 真实样本占主导，并单独强化破坏类负例

默认每批 24 对，其中 18 对真实候选、6 对合成样本。
真实批次按 other/recover/destroy 三类分层采样，因此每批都有真实恢复和真实破坏示例。
初始化难例优先两端原本都是 TP 的邻近候选；训练每 500 步重新挖掘被当前模型高估的 other/destroy 对。
在动作交叉熵之外，加入真实 recover 对与真实 destroy 对的分数排序损失。
记录真实、合成样本**实际生成的类别**；合成生成失败不会将错误类别伪装为期望类别。

准备阶段打印 `newly_recovery_labeled`，用于核对有多少原先被标为 keep 的真实恢复得到了纠正。
所有采样、难例挖掘和点特征归一化只使用 13 个训练森林。验证和 Wytham 不进入训练银行或训练批次。

## 3. 在真实点特征上比较连续高度层

保留 V2 的双向局部点连接编码，新增按实例对共同 z 范围划分的 8 层结构分支。
每层同时池化点特征和表示点数的权重，构造交换实例顺序不变的结构 token。
两层沿高度的一维卷积学习相邻层关系，输出加入最终动作分类。

结构证据包含：

- 同层占据、两端点质量比例、XY 中心与投票中心距离、横向扩展与连接间隙；
- 跨相邻高度层的续接距离、各实例自己的中心轨迹漂移；
- 两端是否各自在连续层有支撑、两条支撑轨迹的间隔，以及下部双轨迹线索。

这些是对抽样点的可观测几何描述，**不是已经验证的树干检测器**。
高度按实例对的 z 范围归一化，未使用地面高度模型。缺层有显式掩码，稀疏层要求最少点数。
模型学习这些证据，不设置“必须有一根树干”或“双轨迹必定不同树”的硬规则。
点特征仍为官方冻结的 32 维缓存；不会重新训练或执行官方骨干网络。

## 源域选策略与局部写出

每 500 步在五个源域验证森林的自然候选上测试预设分数阈值，并保留明确的 keep-all 选项。
候选策略须无 GT 丢失且宏 F1 不低于原版，优先比较恢复数，再比较 F1、覆盖率及更少改动。
另作按森林留出的策略压力检查：用其余四森林选阈值，在被留出森林查看损失；
若出现损失或 F1 下降，该 checkpoint 的最终策略回退到 keep-all。
这仍是源域验证策略分析，不冒充新的独立测试集，不能保证跨域零损失。

部署选择与 V1/V2 一致，使用 mutual-best、单轮、不相交的原实例对。
只将被选择的两个实例取并集，未选择点集和原前景保持不变。无全场重聚类，也不通过 GT 选择部署动作。
评价预测保存后才读取 GT；逐对反事实统计与联合最终指标分别记录。

## 运行方式

V3 复用已完成的 V2 `points.json/points.npz` 与源域 `targets.json/targets.npz`，
需要保留 V1 原候选和官方缓存。新数据只写到 V3 目录，不复制整份冻结点特征，不覆盖 V2。
配置：`configs/experiments/instance_relation_v3/full.yaml`。

```bash
cd ~/projects/zrx/code/TreeLearn
conda activate TreeLearn
git pull --ff-only origin codex/candidate-mask-mlp-step1
bash tools/instance_relation_v3/run.sh all
```

始终前台运行，终端实时显示 doctor → tests → prepare → check → train → evaluate → external。
训练 3000 步；验证每 500 步一次。Wytham 沿用源域最佳 checkpoint 和阈值，不在其 GT 上选阈值。
这次训练的是小型实例关系头，无需重跑官方模型、原始分块或 Wytham 全场骨干推理。

已有 V3 `last.pth` 时续训：

```bash
bash tools/instance_relation_v3/run.sh resume
```

单独复评、外部评价或无 GT 部署：

```bash
bash tools/instance_relation_v3/run.sh evaluate
bash tools/instance_relation_v3/run.sh external
bash tools/instance_relation_v3/run.sh predict
```

日志：`logs/treelearn_instance_relation_v3/latest/`。
输出：`work_dirs/treelearn_instance_relation_v3_s42/`。
结果优先看恢复/丢失及实际接受的 destroy 动作，不能只看 F1 或“流程完成”。

```bash
cat work_dirs/treelearn_instance_relation_v3_s42/train_summary.json
cat work_dirs/treelearn_instance_relation_v3_s42/evaluation/brief.json
cat work_dirs/treelearn_instance_relation_v3_s42/external_evaluation/brief.json
```

`no_edit_result=true` 表示明确回退，不算提升。
每个森林同时保存 `accepted_merges.json`、`action_diagnostics.json` 和 `all_gt_trees.csv`。
`accepted_observed_outcome_counts` 是在当前可见 GT 上逐对重放的结果，和联合 lost/recovered 不必可加。

## 检查范围

新增服务器测试覆盖统一损益优先级、旧纯度门槛不再阻挡恢复、减少 FP 无恢复奖励、
冻结样本复用与失效拒绝、真实样本比例、双下部支撑/垂直续接线索、实例顺序/平移/补零不变性、
动作损失到结构和点特征的梯度、未知连接标签隔离、真实排序损失、单分布策略、
森林留出压力检查、非空局部写出，以及小数据准备/训练/挖掘/续训/评价/无 GT 部署全流程。
本地仅作 Python 3.10 语法、静态接口、配置和 Bash 语法检查，未运行模型或测试。
以上是实现与验证设计，尚无 V3 实验效果。
