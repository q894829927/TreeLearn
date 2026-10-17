# A1 下一步：复用现有缓存的 CPU 失败诊断

日期：2026-10-17。用户已回传完整诊断，详见 [结果与优先级调整](candidate_mask_mlp_failure_audit_results_20261017_zh.md)。以下命令保留用于复现，不必重跑已完成的诊断。本地只做静态检查，没有运行模型、点云或单元测试。

## 1. 本次要回答的问题

1. Wytham 的退化主要关联丢点、改归属，还是二者共同作用？复用已保存预测做两个事后对照，逐候选对账 FP 转移，并记录丢失/恢复 GT 的特定 IoU 变化。
2. 源域有多少有效候选共享同一个 GT 目标？其中实际有多少唯一点同时被多个候选教成正例？仅统计同目标候选数量不够，因为不同区域可能没有共享正例。
3. 训练人口中有多少正例位于 `baseline==0`，部署时永远不参与掩码评分？候选筛选还排除了多少标注不足与充分标注但困难的候选？

这一步不训练 A1b，不新增 attention，不改原权重、归属、阈值或评价定义。训练改动由诊断结果决定，避免同时修改多个机制。

## 2. 读取与保护

入口：`tools/mask_refinement/failure_audit.py`。只使用 NumPy/SciPy 和已有配置/文件，不导入模型或反序列化 checkpoint；checkpoint 只读取字节计算 SHA256 来核对旧锁。主干与 MLP 均不运行，GPU 不参与此诊断。

先验证源域训练签名、manifest、原评价 SHA256、checkpoint SHA256、外部锁和 streaming 实现哈希。再用保存的 Wytham 数组重算 A0/A1 主指标并逐项核对，若不同则停止，不用不一致数据解释原实验。

源域读取原 `audit.json`/`supervision.npz`，只读重算候选监督用于核对，任何不一致都会报错，不重写这些文件。外部数据只临时计算“沿用源域规则会得到什么监督”的事后统计，报告明确标为 `hypothetical_external_policy_not_training`，不会产生外部训练监督文件或参与训练。

按森林、区域及索引块顺序扫描；每个森林额外有一份 int32 点计数数组，用于准确计算唯一共享正例点。统计的是完整候选区域人口，不是重新随机采样。`*_refs` 可重复计同一点在多个候选中的出现次数，`*_unique_points` 对每个森林去重。正例/有效标注的不可部署比例同时提供按引用数汇总与每候选等权平均，避免大区域支配唯一解释。

## 3. 服务器步骤

现有 Wytham `evaluation/summary.json`、`refined_labels.npy`、两域缓存与源域原监督都要保留。先更新并验证代码：

```bash
cd ~/projects/zrx/code/TreeLearn
git switch codex/candidate-mask-mlp-step1
git pull --ff-only
conda activate TreeLearn
mkdir -p logs/candidate_mask_mlp
set -o pipefail

python -m unittest discover -s tests -p 'test_candidate_mask_mlp.py' -v \
  2>&1 | tee logs/candidate_mask_mlp/unit_tests_failure_audit.log
```

诊断初版包含 39 项测试；加入后续 A1b 后，当前共 44 项。诊断新增的 5 项覆盖重复正例与不可部署正例的精确计数/分块一致性、忽略点与候选排除类型、旧监督不一致拒绝、GT 特定 IoU，以及不加载模型、保持原文件字节不变的完整诊断流程。旧 CUDA tile 比较仍在 GPU 可用时执行；诊断本身不使用 CUDA。

测试通过后，后台运行两个域的全部诊断：

```bash
nohup python -u tools/mask_refinement/failure_audit.py --scope all \
  > logs/candidate_mask_mlp/failure_audit.log 2>&1 < /dev/null &
```

与外部后台启动脚本共用同一把锁，准备/评价尚在运行时会拒绝并发。不要同时手工启动不使用锁的准备或评价。日志会在每个森林和每 200 个候选输出进度；需要扫描完整缓存并计算哈希，耗时不等于此前约 40 秒的 MLP 推理。

```bash
tail -n 40 logs/candidate_mask_mlp/failure_audit.log
```

最后出现 `SAVED .../summary_all.json; post-hoc only, no training or prediction.` 才表示该次完整诊断结束。若报错，先提供日志末尾，不重新生成缓存或删旧监督。

## 4. 回传与判读

```bash
cat work_dirs/candidate_mask_mlp_a1_wytham_s42/diagnostics/failure_audit/summary_all.json
```

摘要中主要看：

- `external.delta_vs_baseline`：完整 A1、仅丢点、仅改归属三种结果；对照之间存在交互，不相加解释成独立因果贡献。
- `external.point_changes`、`candidate_state_transitions`：丢点的 GT 构成与新增 FP 的状态来源。
- `source.by_split.train` / `validation`：源域原监督的重复目标、真实共享正例、语义前景排除及候选筛选统计。
- `duplicate_gt_fraction_among_eligible_targets`：分母为有有效候选目标的 GT 数，不是所有 GT 树。
- `shared_positive_foreground_point_fraction`：可部署的唯一正例点中，至少被两个有效候选同时标正的比例。
- `positive_excluded_by_semantics_fraction` / `mean_candidate_positive_excluded_fraction`：正例引用数汇总与每候选等权的不可部署比例；两者不是丢弃点的 GT 比例。

完整源域逐候选统计位于 `diagnostics/failure_audit/source/<plot>/supervision.json`。外部详细报告为同目录下的 `external/supervision.json` 与 `external/assignments.json`；后者包含 `gt_match_changes`，给出每棵丢失/恢复 GT 的 A0/A1 最佳 IoU、最佳候选及假设监督的重复目标情况。它只描述关联，不能证明重复目标导致了该树的丢失。

所有报告保留 `posthoc=true`、`eligible_for_advancement=false` 和输入哈希。若只想分开执行，可用 `--scope external` 或 `--scope source`，输出分别为 `summary_external.json`/`summary_source.json`；原完整评价和源域失败判定保持不变。

下一轮按证据选单变量：若不可部署正例占比显著，优先对齐训练/推理点集；若存在大量可部署共享正例，再优先检验候选唯一性/竞争监督；若原正确候选被频繁改坏，再单独检验保护原归属。不能仅因某个事后对照分数较好，就把它当作新方法或在 Wytham 上选阈值。
