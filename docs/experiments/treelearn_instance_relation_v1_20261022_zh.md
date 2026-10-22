# 官方预测上的局部实例关系头：保持 / 合并 V1

## 本轮实现范围

从官方权重已经生成的对齐缓存出发，原 U-Net、语义、offset、实例点集和输入文件全部只读。新增小型实例关系网络，首轮只决定保持或合并两个原实例。没有再次训练或执行官方 U-Net，不重新生成 tiles、聚类或分配全林点，也不继承已结束的 XY 修正分支。

该模型是官方 TreeLearn 后的可学习局部实例修复，不是新的原始点云端到端骨干。本轮尚无实测增益。已有 Wytham 案例用于解释错误和事后查看；候选生成、部署合并不读 GT，模型与阈值只用既有 13 个训练森林、5 个验证森林确定。

与上一版的直接差别：上一版修改全场 XY，原正确实例中有 581 个点集变化；这一版只有接受的原实例对会合并，未选中实例的编号和完整点集逐点不变。不能事先知道哪一个预测是真 TP，保护质量由网络预测及源域验证评估，不能保证 Wytham 不丢树。

## 输入和候选

只读现有缓存的 `coords.npy`、`baseline.npy`、`votes_xy.npy`、`tree_probability.npy`。不读取大体积 32 维全林特征，不加载官方网络。校验缓存 manifest/signature、官方权重文件名；doctor 校验现有官方权重 MD5。复用缓存的来源信息不是重新证明原训练 / 测试数据独立性。

所有原正实例均保留为节点，不以 GT 排除 unknown-only 或 FP。分块扫描得到完整 bbox、点数和概率统计；按实例自身 bbox 的 4 个 z 层，每层最多抽 64 个实际点，提取 XYZ、原 XY 投票、树概率。抽样依据全局点索引的确定性 hash，与分块尺寸和 GT 无关。GT 负标签不参与任何候选输入。

“下部”指 bbox 内最低有点 z 层的几何代理，不是真实树干标签或地面归一化树高；不强制竖直、单干或树枝完整可见。

候选为两个图的并集：样本投票中心距离 ≤4 m，或下部几何中心 XY 距离 ≤4 m。先计数再分配，超过 200,000 对时直接报错，不静默截断。所有高度层采样也有内存上限。

每节点默认 54 维统计，每实例对 59 维关系统计：

- 节点体量、bbox 形状、概率均值 / 波动、下部位置、分层占比、层内形状与投票离散；
- 实例对 bbox 距离、形状比例、z 重合、点数比例、投票中心差和下部位置差；
- 双向 3D 近邻接触与投票近邻距离，在完整样本及两节点共享的 4 个物理高度层分别统计；
- 所有 pair 特征交换节点后相同，节点特征不编码全局绝对 XY 地址。

统计中的接触距离来自分层抽样，属于近似结构证据；点数、bbox 和训练 / 评价的 GT 交集使用全量缓存。

## 可学习网络与监督

共享节点编码 MLP：54→64→64；对两个节点编码的和、绝对差与 pair 特征共同建模。输出：

1. **merge**：这对是否适合执行有利且保护原正确实例的合并；
2. **same**：两个较纯实例是否主要来自同一棵树；
3. **union quality**：合并掩码的 precision、recall、IoU；
4. **union gain**：合并 IoU 相对两个原掩码最佳 IoU 的增益，由 pair 关系直接回归；
5. **node quality**：各原实例的最佳 GT IoU，用于预测稳定性保护。

默认约 2.5 万参数，准确数量由服务器 `check` / `train_summary.json` 给出。归一化统计只来自训练森林，写入 checkpoint，输入和所有网络损失使用 FP32。

预测特征和候选列表保存完后，才读取训练 / 验证的 `labels.npy`，保存独立 `supervision.npz`。已知点占比至少 20% 才有节点监督；包含 unknown-only 节点的 pair 不作为负例。GT<0 不当背景；GT=0 才是已知背景。

merge 正例需要同一主要 GT、两片纯度至少 0.6、合并 precision 至少 0.65、union IoU 严格 >0.5，且比单片最佳 IoU 增加 >0.02。不把两个原 TP 合为一棵；允许一个原 TP 与同树 FP 碎片作有益补全，但必须满足上述实际质量提升条件，推理时仍受预测稳定性保护约束。GT 只用于训练标签，部署没有读取 TP/FP 标记的接口。相邻真树、会损害原正确实例或引入污染的组合是重要 keep 负例。

损失为 merge BCE + 0.5 same BCE + 2×union quality Smooth L1 + gain Smooth L1 + node quality Smooth L1，union IoU 的回归误差额外加权。增益由 pair 直接预测，避免用两个独立质量估计相减作为唯一的增益判断。实例对按 merge / keep 平衡，再在有该类别样本的森林中均匀抽样；原实例质量按低 / 高 IoU 平衡。没有额外制造 GT 指定的新候选或人为碎片。如果训练缓存完全没有真实有利合并正例，流程明确停止并打印准备统计。

新增头 AdamW，3000 步，batch 256，学习率 1e-3；不解冻官方模型。每 500 步在五个源域验证森林上校准、保存最佳模型。训练和校准只处理小规模节点 / pair 数组，主要一次性开销是 CPU 几何预处理。

## 合并策略与保护

候选必须同时满足预测 merge / same、union precision / recall / IoU 和预测增益要求。预测 node IoU 达到稳定性阈值的原实例锁定。merge 阈值 `[0.6,0.75,0.9,0.97]`、保护阈值 `[0.65,0.8]` 的 8 种组合仅用源域验证选择；另有阈值 1.1 的显式 keep-all 候选。

默认源域允许丢失 GT 数为 0。在符合该要求的策略中，按 macro F1、Coverage、较少编辑依次选择。Wytham 不参与选择。若所有有编辑策略都不可靠，允许保持官方结果，必须按 `no_edit_result=true` 报告，不能声称模型提升。源域选得安全不保证外部数据安全。

对合格 pair 按预测置信度排列，只有双方互为优先的 pair 才接受。一轮内原实例最多参与一次，不做 A–B–C 连锁合并，不对剩余候选重新聚类。

执行只将所选两片中较大原编号映射到较小原编号。未选原实例 / 背景完全不动；前景保持逐点相同。输出 writer 分块检查这两个约束，写入新的 `predictions.npy`，绝不覆盖原 `baseline.npy`。全量 GT 交集表按同样 pair 求和可精确评价 union；服务器回归测试将该结果与真实逐点输出的评价对照。

## 服务器前台执行

本地不运行单元测试、模型前向或训练。源码已经做静态检查，回归测试、实际数据 check 和完整实验交给服务器。

```bash
cd ~/projects/zrx/code/TreeLearn
conda activate TreeLearn
git pull --ff-only origin codex/candidate-mask-mlp-step1

bash tools/instance_relation/run.sh all
```

`all`：doctor → tests → prepare（18 个源域森林）→ 实际 pair 梯度 / 对称性 check → train → 内部验证 → Wytham 特征提取、预测与评估。全过程前台显示，测试失败即停止；不在后台启动。共享旧森林实验的 flock，避免同时启动流程。

输出目录：

- 几何与监督：`data/treelearn_instance_relation_v1_s42/{train,validation,external}/{plot}/`；
- 模型 / 校准：`work_dirs/treelearn_instance_relation_v1_s42/`；
- 日志：`logs/treelearn_instance_relation_v1/latest/`。

每个森林打印节点数、候选对数、真实有利合并正例数；预处理可复用已完成文件。训练打印 loss 和验证选择的策略。准备阶段中断、尚未产生 `last.pth` 时重新运行 `all`；产生 checkpoint 后继续：

```bash
bash tools/instance_relation/run.sh resume
```

保持代码 / 配置不变；新版本来源签名只覆盖本模块，不更改旧实验的签名文件。仅重复评估已有最佳头：

```bash
bash tools/instance_relation/run.sh evaluate
bash tools/instance_relation/run.sh external
```

单独部署、不打开 GT、不制作监督：

```bash
bash tools/instance_relation/run.sh predict
```

`predict` 读取 external cache 的预测输入与元数据，允许完全没有 `labels.npy`。`external` 则先保存预测，随后单独打开 GT 评价。已有完整预测会校验模型、策略、特征与文件 hash 后复用；未完成的自有输出有 request 记录时可重建，不覆盖不同输入的结果。

## 看哪些结果

```bash
cat work_dirs/treelearn_instance_relation_v1_s42/train_summary.json
cat work_dirs/treelearn_instance_relation_v1_s42/evaluation/brief.json
cat work_dirs/treelearn_instance_relation_v1_s42/external_evaluation/brief.json
cat work_dirs/treelearn_instance_relation_v1_s42/external_evaluation/wytham_vox0.1/metrics.json
cat work_dirs/treelearn_instance_relation_v1_s42/external_evaluation/wytham_vox0.1/watchlist.json
```

重点看 accepted_pairs、F1、TP/FP/FN、lost/recovered、Coverage，而不是仅训练 loss。`prediction.json` 给出 `changed_foreground_points=0`、`changed_unselected_points=0`，以及所选原实例数；`accepted_merges.json` 记录每个合并动作。`metrics.json` 另报原正确实例被选择 / 完整保留的数量，及 GT 辅助的候选正例覆盖；`candidate_recovery_oracle_gt_ids` 单列原漏检的可合并目标，区别于已有 TP 的有益补全。这些 GT 信息只作事后诊断。

六个已有碎片案例在 watchlist 中直接比较，但不能将这六例当作随机独立测试集。当前方法仅合并，不能从原 0 点找回 129 棵前景缺失树，也不能拆开已经混合的实例。使用同一已反复分析的 Wytham 仍属于外部数据开发评估。

回归用例覆盖 GT 不参与特征或部署、分块采样一致、pair 对称性、密集候选内存保护、unknown 监督隔离、union 计数与逐点评价一致、互为优先且不连锁、未选点集不变、keep-all 复现、归一化 checkpoint 和小流程训练 / 推理。它们由上述服务器 tests 阶段执行，本地没有执行。

## 首轮结果与保存结果诊断

3000 步训练结束，源域选择 2500 步 checkpoint，merge 阈值 0.6、保护阈值 0.65。用户服务器结果：

| 数据 | 接受合并 | 找回 / 丢失 GT | F1 变化 |
| --- | ---: | ---: | ---: |
| 5 个源域验证森林 | 5 | 0 / 0 | +0.1257 个百分点 |
| Wytham | 30 | 0 / 5 | −0.0928 个百分点 |

Wytham TP/FP/FN 从 596/222/281 变为 591/215/286。未选实例与前景逐点不变，但合并选择涉及 12 个原 TP，最终全场丢失 5 棵原检出树；总指标尚未给出这两组的逐一对应，需由动作回放核对。因此这轮没有实现找回漏检的目标，不能把 FP 减少或源域 F1 微增说成外部修复成功。

已有事实和待验证假设需分开：训练 merge 正例同时包含“原 TP 加同树碎片的有益补全”和“两个碎片合并找回原漏检”；源域这次只体现了前者或误检清理的收益。保护阈值检查的是预测 IoU ≥0.65，真正 TP 的条件却是 IoU >0.5。实际 IoU 在 0.5～0.65 的原正确实例即使预测准确，也仍允许参与合并；实际高质量却预测低于 0.65 则是另一种失败。哪些属于哪一种，及已知六组可合并碎片为何未恢复，需要读取保存的分数和交集表，不能仅凭总指标认定特征坍缩或候选不足。

新增 CPU 诊断入口只读已保存的 `features.npz`、`supervision.npz`、`scores.npz` 与结果元数据；不修改训练配置、模型、策略或预测，不执行模型前向，不重新扫描 3700 万点。它保留原训练代码签名，可直接审计已有 checkpoint：

```bash
cd ~/projects/zrx/code/TreeLearn
conda activate TreeLearn
git pull --ff-only origin codex/candidate-mask-mlp-step1

python -m unittest discover -s tests -p 'test_instance_relation_diagnostics.py' -v &&
python -u tools/instance_relation/diagnose.py
```

输出保存在 `work_dirs/treelearn_instance_relation_v1_s42/relation_diagnostics/wytham_vox0.1/`：

- `brief.json`：各训练森林正例数、候选覆盖、逐道门槛剩余数量、保护阈值设计缺口与预测低估数量、丢失 / 找回 ID；
- `accepted_pair_audit.json`：每个已接受 pair 的预测 / 实际 union 质量、原实例质量、单独执行该合并时的完整匹配评价；
- `watchlist.json`：六组已知碎片是否进入候选、全部未通过的门槛、是否被互选规则排除、对应 GT 的实际 union IoU 与独立匹配结果。

```bash
cat work_dirs/treelearn_instance_relation_v1_s42/relation_diagnostics/wytham_vox0.1/brief.json
cat work_dirs/treelearn_instance_relation_v1_s42/relation_diagnostics/wytham_vox0.1/watchlist.json
```

候选中的 `potential_recovery` 只代表合并后有 IoU >0.5 的可能匹配，且原 TP 未丢失其 IoU 匹配，不是最终检测数或可部署收益。现有协议先做最大总 IoU 的 Hungarian 分配，再按严格 >0.5 筛选，因此不能把每个局部 IoU 超阈值都直接计作找回。报告中每个已接受动作和六组关注 pair 都重新执行完整匹配，并将所有动作一起执行的结果与原保存指标核对。保留原评价协议，不为这轮修改指标。

诊断用例覆盖实际逐点评价一致性、合并两个真树的损害、保护门槛筛除、已知案例的追踪、IoU 可行但 Hungarian 未检出的反例、分数 / 动作不一致和空候选。仍由服务器执行，本地只做静态检查。诊断完成后再确定是否修改候选、关系学习或保护策略；不直接延长训练，也不根据 Wytham GT 反选部署动作。
