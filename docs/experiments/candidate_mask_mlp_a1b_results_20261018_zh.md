# A1b 原归属候选排序监督：源域负结果

日期：2026-10-18。来源：服务器回传的完整五森林 `evaluation/summary.json` 和 `training_summary.json`。本地只解析 JSON、核对汇总和代码哈希，没有运行模型、点云或单元测试。

## 1. 判定与比较条件

**A1b 未通过预设条件；当前结果不支持采用这版 owner 排序损失。** Coverage 达到预设的 +0.5 pp，但 F1 下降、Commission 增加超过允许范围。相对上轮 A1，Coverage 小幅提高，F1 和 Commission 继续恶化，不能将局部 IoU 收益表述为整体分割改进。

本轮沿用 A1 的模型、缓存、监督映射与主候选筛选、抽样/权重、归一化、seed 42、训练预算、按原验证 BCE/Dice 选模及推理规则，只增加 `ownership_rank_weight=1.0` 的训练项；详见 [预定协议](candidate_mask_mlp_a1b_owner_rank_20261017_zh.md)。两轮完整评价的每森林 baseline 对象逐项相同，以下是同缓存、同协议下的比较。

这是单个 seed、一个预设权重和一种配对方式的结果，不能据此否定全部竞争监督、MLP 或 query-mask 方法。当前没有 A1b Wytham 结果。

## 2. 五森林等权宏平均

| 指标 | A0 | A1 | A1b | A1b−A0（pp） | A1b−A1（pp） |
|---|---:|---:|---:|---:|---:|
| F1 | 88.195% | 87.600% | 86.539% | -1.656 | -1.061 |
| Completeness | 91.164% | 91.453% | 91.636% | +0.472 | +0.183 |
| Commission，越低越好 | 14.399% | 15.719% | 17.758% | +3.359 | +2.039 |
| Coverage | 82.076% | 82.384% | 82.638% | +0.561 | +0.254 |
| Point precision | 89.349% | 90.982% | 91.178% | +1.828 | +0.196 |
| Point recall | 90.581% | 88.297% | 88.361% | -2.220 | +0.064 |

Coverage、Completeness、至少 3/5 森林 Coverage 不下降三项通过；F1 和 Commission 两项失败，保持原报告 `passed=false`。Point precision/recall 为每 GT 选择最高 IoU 预测后计算再平均的指标，并非全森林树/背景 micro 指标。

| 森林 | A1 ΔF1（pp） | A1b ΔF1（pp） | A1b ΔCoverage（pp） | A1b ΔCommission（pp） | A1b 原前景未分配比例 |
|---|---:|---:|---:|---:|---:|
| G4N | +1.150 | +0.589 | +1.667 | +0.379 | 3.746% |
| G4W | +0.026 | +0.142 | +1.195 | +0.537 | 2.550% |
| L1N | -0.746 | -2.872 | +0.148 | +4.583 | 1.994% |
| O1N | -2.044 | -3.641 | -0.235 | +6.888 | 1.839% |
| O1W | -1.359 | -2.496 | +0.031 | +4.408 | 3.646% |

各项差值相对各森林 A0，只有 A1b−A1 明确标识的列例外。A1b 的 4/5 森林 Coverage 上升，但只有 G4N/G4W 的 F1 上升；Commission 五处都增加。尤其 O1N 的 TP 仍为 152，FP 从 11 增至 24，F1 下降不能仅由“基线高”解释。

## 3. 实例计数与点变化

| 五森林合计，非 macro 指标 | A0 | A1 | A1b |
|---|---:|---:|---:|
| GT 数 | 1782 | 1782 | 1782 |
| TP | 1633 | 1639 | 1642 |
| FP | 275 | 300 | 345 |
| FN | 149 | 143 | 140 |
| 非空预测候选数 | 2595 | 2425 | 2469 |
| 原前景未分配点 | 0 | 1188829 | 1775294 |
| 原前景未分配比例 | 0% | 1.851% | 2.765% |
| 跨候选改归属点 | 0 | 1345511 | 1406964 |

A1b 相对 A0 的 TP 净增 9、FP 净增 70；相对 A1 仅多 3 个 TP，却多 45 个 FP。恢复原未匹配 GT 21 棵、丢失原已匹配 GT 12 棵，净 +9 与 TP 对账。候选数是另一种统计对象，不能据其减少 126 个推断误检控制改善。

64,213,230 个原前景点全部参与评分，各森林预测非空、没有未评分前景，排除了 smoke 阶段的“全低于阈值”故障。A1b 总改动 3,182,258 点，其中未分配 1,775,294 点、改归属 1,406,964 点；相对 A1 未分配点多 586,465（+49.33%），改归属多 61,453。五处未分配比例均高于 A1。

尚没有 A1b 的逐候选状态转移或被丢弃点的 GT 构成，因此不能把新增 70 个 FP 全部叫作新造错误实例，也不能把全部未分配点叫作被误删树点。需要分开检查原已匹配候选受损、原不计 FP 候选变为计数 FP，以及原 FP 退出计数的情况；原 A1 的 52/64 比例不能套用到 A1b。

## 4. 从本轮能学到什么

1. **当前排序项没有产生预期的整体改善。** 受控变更后 Coverage 上升，但误检代价更大。这既不证明所有相对竞争监督无效，也不支持立即把该项叠加到更复杂模型上。
2. **相对排序不能单独保证绝对分数过阈值。** 当前项只约束两个 logit 的差；例如正确候选 -1、错误候选 -2 的相对排序可以更好，但两者 sigmoid 都低于 0.5。原 BCE/Dice 仍提供绝对监督，最终结果由联合优化决定。未分配量增加与这种局限相容，但不能据此断言它就是本轮退化的已证实原因。
3. **点集 IoU、覆盖率和实例级误检需要同时看。** 4/5 Coverage 提高、全部 Point precision 提高，并不保证 IoU > 0.5 匹配后的 F1 提高；选定的局部验证损失也不等同于完整实例指标。
4. **原 owner 配对并非完整的多候选竞争。** 它跳过同 GT 目标、背景/未知和双方都不正确的点，owner 目标仍来自原冻结映射；它没有显式处理全部竞争者或增加实例质量监督。这是方法边界，不是已经证实的主因。

下一步只复用 A1b 保存预测做一次 CPU 操作对照与 FP 转移核算，信息目标是解释本轮额外误检的来源。该诊断不改变主判定；其对照是事后分析，不能代替预先定义的新方法。完成后再决定是否另立单变量实验，保留当前权重、缓存和负结果。

## 5. 服务器诊断步骤

已有 `run.py diagnose` 支持独立配置，不需要修改模型、重新训练或重跑主干。当前三个 A1b 代码签名文件保持不变。

```bash
cd ~/projects/zrx/code/TreeLearn
git switch codex/candidate-mask-mlp-step1
git pull --ff-only
conda activate TreeLearn
mkdir -p logs/candidate_mask_mlp_a1b_owner_rank

nohup flock -n logs/candidate_mask_mlp/background/external.lock \
  python -u tools/mask_refinement/run.py diagnose \
  --config configs/experiments/candidate_mask_mlp/step1b_owner_rank.yaml \
  > logs/candidate_mask_mlp_a1b_owner_rank/diagnose.log 2>&1 < /dev/null &
```

这里复用现有外部/训练锁，避免和仍在运行的训练或诊断并发。若未得到锁，先确认现有任务状态；不删除锁文件绕过正在运行的任务。诊断不更新原训练 runner 的 `status.txt`。

查看进度与完成后回传：

```bash
tail -n 20 logs/candidate_mask_mlp_a1b_owner_rank/diagnose.log

# 日志出现 SAVED .../diagnostics/assignment/summary.json 后：
cat work_dirs/candidate_mask_mlp_a1b_owner_rank_s42/diagnostics/assignment/summary.json
```

需要的三个结果为：仅丢点/仅改归属的宏平均变化、FP 净增 70 的状态转移对账、未分配点中 GT 树点/背景点的数量。A1 的既有诊断无需重跑；本次没有安排 Wytham 或新的训练。

## 6. 训练与溯源

- 代码提交：`9b91baa1975b2c695beafb4621de50b556b99f1d`。
- 参数数 19,649，训练/验证有效候选 5123/1919，排序权重 1.0；未增加模型参数。
- `best_validation_loss=0.17096177538931862`，上轮 A1 为 `0.16839134637032666`；差 +0.002570429，两者均为原验证 BCE/Dice。尚未收到本轮 history，不能报告 best epoch、训练排序损失曲线或实际轮次耗时。
- 训练 CUDA 峰值 85,067,264 字节（81.126 MiB），A1 为 62,022,656 字节（59.149 MiB），本轮增加约 37.2%；不是等算力比较。当前没有完整流水线 RAM、训练时长或逐项单元测试日志，不补写这些数值。
- 评价附件 SHA256：`3cdc2695b15ae1f15550e1ff5c0716a7f2d39e38ab87855170c8f1ed2e64d4cb`。
- 评价语义 digest：`85773fe418f2ca74c81dcd2a1d5079e6d207aebfe9fdcf01752fc6aa9633d705`，按 `sha256(json.dumps(result, sort_keys=True).encode())` 计算。
- 用户训练摘要的三个代码 SHA256 与 Git 提交中 LF 内容逐项相同：
  - `tools/mask_refinement/run.py`：`f3045cbbe30eda86c07752f68c72b23d7c5aad18dcd1bf76b4841245a8ab821a`。
  - `tree_learn/mask_refinement/model.py`：`8f5ac4888bc2585322cd330231fcd695699b4766fccf7867ac2ec81df4994a39`。
  - `tree_learn/mask_refinement/ownership_rank.py`：`bfed145a6e9d43ddfff5b022db0cbf81a36b5255f14d7eb601e5c2de7009e55d`。

评价的宏平均、每森林 matched GT 集合的丢失/恢复以及两轮 baseline 一致性已在回传 JSON 上核对。本地没有重算点云指标或执行诊断；当前仍待服务器诊断回传。
