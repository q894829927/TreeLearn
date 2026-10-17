# A1b：原归属候选的相对排序监督

日期：2026-10-17。代码已实现，待服务器运行；本地只做静态检查。依据 [CPU 诊断回传](candidate_mask_mlp_failure_audit_results_20261017_zh.md) 选择此单变量实验，不声称已定位唯一主因或已取得提升。

## 1. 固定什么，改变什么

沿用 A1 的原始特征缓存、所有候选与区域、冻结 GT 目标映射和有效候选集合、原抽样/反采样权重、32 维归一化统计计算、MLP（19,649 参数）、seed 42、40 epochs、batch 4、4096 点/候选、AdamW/学习率、0.5 阈值及最终 argmax 分配。MLP 从相同 seed 重新初始化，不从失败的 A1 best.pth 微调。

唯一方法变量为训练损失：`L = 原 BCE + 原 Dice + 1.0 × owner 排序损失`。没有改为原前景专用采样，也没有在推理中加 owner margin、回填、去重、合并、质量筛选或 attention。原模型结构和推理代码不变。

同一批候选、同一批点和相同 epochs 并不表示 FLOPs 相等：排序项需额外计算原 owner 的分数。训练时间、RSS 与既有 CUDA 峰值记录应一并报告，不能宣称等算力收益。

新配置：`configs/experiments/candidate_mask_mlp/step1b_owner_rank.yaml`。缓存根仍为 `data/candidate_mask_mlp_a1`；新工作目录为 `work_dirs/candidate_mask_mlp_a1b_owner_rank_s42`，原 A1 权重、预测、监督与外部锁继续保留，无需重跑 prepare/audit。

## 2. 排序标签

对原抽样得到的点 p，当前候选为 c，原归属候选为 o。各候选的 GT 目标 t 使用原冻结 audit 的 `target_ids`；owner 包含未被选为训练主候选的 ID，不使用新的 GT 匹配或放宽原主候选筛选。

- GT(p)=t(c) 且不等于 t(o)：要求 `s(c,p) > s(o,p)`。
- GT(p)=t(o) 且不等于 t(c)：要求 `s(o,p) > s(c,p)`。
- c=o、GT 为背景/未知、原语义背景点、两候选都对应该 GT 或两者都不对应该 GT：排序项忽略，原 BCE/Dice 仍按旧规则计算。

其中 s 是同一个 MLP 的 logit。排序采用 `softplus(-y × (s(c,p)-s(o,p)))`，y 为 +1/−1；使用原采样权重，在每候选的可比较点上归一化，再对候选等权平均。无可比较点的候选排序损失为 0；整轮没有可比较点则明确报错。

owner 点输入由原 `point_inputs` 按 owner 的中心、尺度、投票中心、原成员标记计算；描述为 owner 的原 64 维描述。特征仍不读 GT，GT 只产生损失标签。原成员一定在自己的区域内，因此所比较的 owner 分数对应实际推理时可用的分数。两个分数均参与反向传播，无新参数。

这是对冻结候选 GT 映射下的竞争一致性进行监督，不是 GT 直接决定部署归属。原候选映射本身仍可能不完美，同 GT 重复候选仍然存在，丢点问题也未被强制修复；是否泛化需实验判断。

## 3. 选模和判定预先固定

训练记录联合 `train_loss`，并分开记录 `train_mask_loss`、`train_rank_loss`、`rank_positive_draws` 和 `rank_negative_draws`。draws 允许重复采样，不是唯一点数。

**best checkpoint 仍由原固定验证采样的 BCE/Dice 宏平均选择**，不会改为联合损失选模；验证采样仍使用原 RegionDataset 与 seed 1907。完整森林 gate 仍相对 A0：Coverage 至少 +0.5 pp、F1 不下降、Completeness 下降不超过 0.5 pp、Commission 增加不超过 0.5 pp、至少 3/5 森林 Coverage 不下降。只超过旧 A1 但没有超过 A0，不算通过。

新签名额外绑定排序模块、MLP 模块和训练驱动的代码 SHA256；修改后不能混用旧 A1b checkpoint 继续 resume/evaluate。原配置没有排序项，其旧签名算法保留，原 A1 的评价与诊断继续可用。

本次只进行源域实验。若源域失败，完整保留结果后再分析，不自动测试 Wytham 或进入 attention。方案已受 Wytham 诊断启发，即使后续报告 Wytham，也属于已观察数据上的探索结果，需要另外的独立验证数据支持稳定性。

## 4. 服务器一键执行

```bash
cd ~/projects/zrx/code/TreeLearn
git switch codex/candidate-mask-mlp-step1
git pull --ff-only
conda activate TreeLearn
mkdir -p logs/candidate_mask_mlp_a1b_owner_rank

nohup bash tools/mask_refinement/run_owner_rank.sh \
  > logs/candidate_mask_mlp_a1b_owner_rank/launch.log 2>&1 < /dev/null &
```

脚本按顺序执行：44 项单元测试 → 独立 smoke 训练 → 完整 40 epoch 源域训练 → 五森林完整评价。每阶段成功才继续；不启动任何主干推理/缓存生成或 Wytham 评价。smoke 仅验证损失、排序样本、保存/加载流程，不用其分割指标判断有效性。

新增 5 项测试覆盖同一点配对/跳过重复目标、双分数梯度方向、空配对、配置及旧签名兼容、原采样不变和排序训练的 smoke/resume/evaluate 集成。旧 CUDA tile 比较会在 CUDA 可用时运行。

运行中查看：

```bash
cat logs/candidate_mask_mlp_a1b_owner_rank/status.txt
tail -n 30 logs/candidate_mask_mlp_a1b_owner_rank/launch.log
tail -n 10 logs/candidate_mask_mlp_a1b_owner_rank/train_full.log
```

尚未进入 full 阶段时，`train_full.log` 不存在，按 status 查看 `unit_tests.log` 或 `smoke.log`。阶段耗时/RSS 在同名 `.time.txt`，退出码在 `.exit_code.txt`。脚本与外部运行/诊断共用锁，避免并发；若已存在 A1b checkpoint 则拒绝新启动，不覆盖它。

最终状态 `DONE source_gate_passed` 或 `DONE source_gate_failed` 均表示源域评价已完成；后者伴随评价退出码 2，是未过预设条件的科学结果。`FAILED <stage>` 则先检查该阶段日志和是否生成完整报告。

完成后回传：

```bash
cat work_dirs/candidate_mask_mlp_a1b_owner_rank_s42/training_summary.json
cat work_dirs/candidate_mask_mlp_a1b_owner_rank_s42/evaluation/summary.json
```

若中断在 full 训练阶段且新工作目录已有 `last.pth`，可在同版本代码上用既有 CLI 单独恢复，不重跑整套脚本：

```bash
nohup python -u tools/mask_refinement/run.py train \
  --config configs/experiments/candidate_mask_mlp/step1b_owner_rank.yaml --resume \
  > logs/candidate_mask_mlp_a1b_owner_rank/train_resume.log 2>&1 < /dev/null &
```

确认恢复完成后，再单独执行 `run.py evaluate --config configs/experiments/candidate_mask_mlp/step1b_owner_rank.yaml`。恢复前检查没有仍在运行的同一任务；不删除 A1/A1b 缓存或权重来绕过保护。
