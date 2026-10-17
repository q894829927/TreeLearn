# 第 ① 步：固定 TreeLearn 候选的条件 MLP 掩码细化

日期：2026-10-15。代码分支：`codex/candidate-mask-mlp-step1`。

状态：完整服务器实验与事后诊断均已回传，当前 A1 未通过源域预设条件，详见 [源域结果](candidate_mask_mlp_a1_results_20261016_zh.md)。只丢弃点、只改归属两个对照的 F1 也均下降。用户随后要求的固定权重 [Wytham 跨域评价](candidate_mask_mlp_wytham_zh.md) 亦已完成，F1 -3.658 pp、Coverage -3.214 pp，详见 [外部结果](candidate_mask_mlp_wytham_results_20261017_zh.md)。当前版本结束追加试验，没有进入 attention 或启动新训练。以下命令保留用于复现，无需重复运行；本地未运行单元测试、点云推理、数据生成或训练。

下一步已提供 [复用现有缓存的 CPU 失败诊断](candidate_mask_mlp_failure_audit_20261017_zh.md)，先量化竞争目标、部署点集和 Wytham 两种操作的作用，再决定单变量训练改动。

诊断已回传并记录，下一轮为 [A1b 原归属排序损失](candidate_mask_mlp_a1b_owner_rank_20261017_zh.md)：只改训练目标，模型/候选/采样/推理保持相同；代码已实现，尚待服务器实验。原 A1 负结果和原文件继续保留。

## 1. 本轮只验证什么

使用官方 `model_weights_with_small_20241213.pth`，关闭 upper-anchor、axis、height adapter、U-Net enhancement、seed confidence 和实例质量过滤。采用 legacy anchor、单 anchor 分组和现有默认 HDBSCAN。它是本仓库当前官方权重推理基线，不等同于重新复现 2024 年论文全部训练与 DBSCAN 设置。

先一次性生成原版预测及 32 维主干特征，再训练候选条件 MLP。主干、原 heads、候选数及候选区域均不参与训练。没有新增注意力、查询补充、拆分、合并或质量排序头。

每个候选的 XY 包围框外扩 2 m，保留全高度。区域、候选中心、尺度、投票中心、Mean/Max 特征只使用预测。候选描述为 64 维；逐点输入为 40 维（32 维主干特征＋相对坐标 3＋相对投票 2＋树概率 1＋当前成员标记 1＋相对高度 1）。条件 MLP 隐藏维度为 64，输出成员 logit，损失为 BCE＋Dice。

原近邻分配在 base-vote XY 空间进行。本轮从该完整候选结果出发，学习候选区域内的点归属，额外使用实际空间坐标及主干特征。

## 2. 输入与标签的隔离

- `prepare`：运行冻结 TreeLearn，导出逐点缓存；建立预测几何候选区域。
- `audit`：单独读取 GT，计算候选匹配、有效监督和 GT 掩码 Oracle。
- `train`：匹配 IoU 至少 0.1、候选纯度至少 0.5、已标注比例至少 0.8 的候选才用作正样本；完全没有 GT 树交集且已标注充分的候选学习空掩码。其余歧义候选不用于监督，但全部参与推理。
- `evaluate`：先调用完全不读取 `labels.npy` 和 `supervision.npz` 的推理函数，预测返回后才读取 GT 算指标。

训练使用正点、邻树难负点和背景点分层采样，并用逆采样权重恢复各类在原区域中的占比。GT=-1 的点不进入损失和评价。特征标准化只拟合训练森林。

本轮固定原始语义前景：原先判为非树的点保持 0。原先判为树的点在所有覆盖它的候选之间比较成员概率，阈值固定为 0.5；没有通过阈值的候选时，点被记录为未分配，输出标签为 0。没有强制近邻兜底。完全相同的分数优先保留原归属，其次选择较小的候选 ID。

因此，本轮不能恢复原始语义漏检，也不能为没有候选的树凭空新增实例。没有显式的质量删除，但掩码为空或竞争失败仍可能使候选消失，结果会报告这类变化。

## 3. 服务器准备与测试

以下所有命令都从服务器的 `TreeLearn` 仓库根目录执行，不能停留在 `TreeLearn/data` 目录。不要覆盖服务器未提交修改。以下命令假设目标分支此前尚未在服务器创建：

```bash
cd ~/projects/zrx/code/TreeLearn
git fetch origin
git switch --track origin/codex/candidate-mask-mlp-step1
conda activate TreeLearn
mkdir -p logs/candidate_mask_mlp
set -o pipefail

python -m unittest discover -s tests -p 'test_candidate_mask_mlp.py' -v \
  2>&1 | tee logs/candidate_mask_mlp/unit_tests.log
```

若分支已经存在，则使用 `git switch codex/candidate-mask-mlp-step1` 和 `git pull --ff-only`。使用 `discover` 按文件发现测试，避免无 `__init__.py` 的 `tests` 目录与环境中的同名包发生导入冲突。测试包含合成数据上的训练、保存/恢复和推理闭环，使用 CPU，不读取真实森林，不需要 spconv。测试必须通过后再生成真实缓存。

检查 [配置](../../configs/experiments/candidate_mask_mlp/step1.yaml) 中两项路径：

```yaml
source_root: data/train/forests
checkpoint: data/model_weights/model_weights_with_small_20241213.pth
```

训练森林为 A1N/A1W、G1N/G1W、G2N/G2W、G3N/G3W、L2N/L2W、LG1/LG2/LG3；验证森林为 G4N、G4W、L1N、O1N、O1W。同名森林只允许一个受支持输入文件，避免混用不同版本。

只增加 NumPy/SciPy/PyYAML/PyTorch 代码，未引入新依赖。`prepare` 仍需服务器现有完整 TreeLearn/spconv 环境。缓存采用独立 `.npy` 和内存映射，训练不会将全部森林复制到 GPU。逐点缓存约 `168 × 点数` 字节，区域索引另需约 `8 × 区域点引用总数` 字节；还需保留原 pipeline 的切块与预处理文件。先用 pilot 检查磁盘与内存需求。

## 4. 先执行 pilot

Pilot 仅使用 A1N 和 G4N。每步退出成功再运行下一步：

```bash
python -u tools/mask_refinement/run.py prepare --pilot \
  2>&1 | tee logs/candidate_mask_mlp/prepare_pilot.log

python -u tools/mask_refinement/run.py audit --pilot \
  2>&1 | tee logs/candidate_mask_mlp/audit_pilot.log

python -u tools/mask_refinement/run.py train --pilot --smoke \
  2>&1 | tee logs/candidate_mask_mlp/train_smoke.log

python -u tools/mask_refinement/run.py evaluate --pilot --smoke --export \
  2>&1 | tee logs/candidate_mask_mlp/evaluate_smoke.log
```

`prepare` 对每棵森林重新计算权重、源点云和关键 pipeline 源码的指纹，已完成且指纹一致的缓存可以续用。中断时重新执行同一命令。单森林具体进度见 `data/candidate_mask_mlp_a1/logs/A1N.log` 等文件。几何区域过大时会报错，不会静默截断。

Smoke 每森林最多取 16 个有效候选训练 1 epoch，写入独立的 `work_dirs/candidate_mask_mlp_a1_s42/smoke/`，只检查闭环。Smoke 评价会推理 G4N 的全部候选；其指标没有性能解释意义，`passed` 为 null，不能用于选择方法或参数。

如 smoke 训练中断或需要继续查看结果，已存在 checkpoint 时加 `--resume`；不需要重跑完成的训练。

Pilot 结束优先提供：

- `data/candidate_mask_mlp_a1/pilot_audit.json`
- `work_dirs/candidate_mask_mlp_a1_s42/smoke/training_summary.json`
- `work_dirs/candidate_mask_mlp_a1_s42/smoke/evaluation/summary.json`
- 如失败，提供对应日志末尾与错误栈。

### Smoke 输出全空时的诊断

本次服务器反馈：19,649 个参数，训练/验证各 16 个候选，1 epoch 后验证损失为 1.294315；G4N 的 364 个候选全部消失，11,167,502 个原树点全部未分配。这说明流程运行完成，但不能据此认定预测正常。默认 batch size 4 对应仅 4 次参数更新，训练不足是一种可能，原输出没有记录概率范围，尚不能确认原因。零预测时 Commission 为 0 也不表示误检控制改善。

推理新增 `assignment.score_diagnostics`，只记录统计，不修改模型、阈值、竞争规则、缓存或 checkpoint 格式。更新代码并运行测试后，使用现有 smoke checkpoint 重新评价即可，无需重新 `prepare`、`audit` 或 `train`，也不必再次导出大体积 NPZ：

```bash
python -u tools/mask_refinement/run.py evaluate --pilot --smoke \
  2>&1 | tee logs/candidate_mask_mlp/evaluate_smoke_diagnostics.log
```

查看 `max_probability`、`num_pairs_at_or_above_threshold`、`num_foreground_points_not_scored` 和 `empty_prediction_reason`。统计中的 pair 是“候选—点”组合，重叠区域里的点会被多个候选评分，所以 pair 数通常大于唯一点数。若所有前景点均已评分、最大概率小于 0.5 且原因是 `all_scores_below_threshold`，可确认当前 checkpoint 的掩码没有通过固定阈值；这仍不能区分训练不足与其他学习问题。若有前景点未评分，先排查区域/缓存覆盖；出现 `assignment_inconsistency` 则先排查分配实现。不要根据 smoke 验证结果降低阈值或添加回退分配来消除空输出。

诊断与完整数据审计正常后再进行正式训练。正式模型能否有效改善点归属，由预设的五森林评价条件判断。

## 5. 完整数据与可恢复性检查

Pilot 闭环通过后，执行：

```bash
python -u tools/mask_refinement/run.py prepare \
  2>&1 | tee logs/candidate_mask_mlp/prepare_full.log

python -u tools/mask_refinement/run.py audit \
  2>&1 | tee logs/candidate_mask_mlp/audit_full.log
```

可复用已经生成的两个 pilot 森林。完整 manifest 必须恰好包含配置指定的 13/5 森林。

训练前的预设可行性条件：

- 训练集至少 500 个有效正候选；验证集至少 100 个；
- 五森林平均 Oracle Coverage 相对同缓存 baseline 至少存在 1.0 pp 的空间。

Oracle 在相同候选区域和原语义前景内，用 GT 选择纯净目标掩码，并做一对一候选—GT 分配以限制每个候选只有一个目标。它是候选与区域能力的诊断参考，不是可部署预测，也不是“这个 MLP 必定能达到”的性能上限证明。其 Completeness 是最大 Coverage 分配下的值。

如果 `audit.json` 的 `passed=false`，命令返回 2，训练也会拒绝启动。先检查候选覆盖、输入标签、监督比例与 Oracle 结果，不应为了继续训练而降低门槛。已有缓存与新配置不兼容时，请使用新的 `output_root`，不覆盖旧实验。

## 6. 正式训练及断点续训

```bash
python -u tools/mask_refinement/run.py train \
  2>&1 | tee logs/candidate_mask_mlp/train_full.log
```

默认 seed 42、40 epochs、batch size 4、每候选 4096 个采样点、AdamW。只训练小型 MLP，所有 backbone 特征均来自缓存。

每 epoch 保存 `last.pth`。按五个验证森林等权平均的固定采样 BCE＋Dice 选择 `best.pth`，不使用 Wytham、不扫成员阈值。完整森林验证只在训练完成后执行，采样验证损失不等同于最终单木指标。

中断后：

```bash
python -u tools/mask_refinement/run.py train --resume \
  2>&1 | tee logs/candidate_mask_mlp/train_resume.log
```

训练设置、输入 manifest、监督规则和推理阈值均与 checkpoint 绑定；改变它们需开启独立实验。每 epoch 的候选顺序和采样由 seed/epoch 确定。显卡算子仍可能有数值差异，不能将单随机种子视为稳定增益证据。

## 7. 完整五森林评价

```bash
python -u tools/mask_refinement/run.py evaluate --export \
  2>&1 | tee logs/candidate_mask_mlp/evaluate_full.log
```

`--export` 额外输出全局坐标的 baseline/refined NPZ，便于可视化；如果磁盘紧张可以省略，标签与未分配 mask 仍会保存。

输出目录为 `work_dirs/candidate_mask_mlp_a1_s42/evaluation/`：

- `summary.json`：五森林 macro baseline、refined、差值、Gate 和各森林结果；
- `metrics.csv`：可直接整理成表格；
- `<forest>/metrics.json`：恢复/丢失 GT ID、消失候选、未分配树点等；
- `<forest>/refined_labels.npy` 与 `unassigned_tree_mask.npy`：与缓存行号严格一致；
- 开启导出时的 `<forest>/baseline.npz` 与 `refined.npz`：`points`＋`labels`。

评价协议明确限定为对齐的 voxelized cache：忽略负 GT 标签，IoU 严格大于 0.5 才匹配；未匹配预测至少一半点属于标注树时才计 Commission；Coverage 对所有 GT 树平均，完全漏分记 0。A0/A1 在同一缓存和同一协议下比较；不能与过去 original-resolution/Wytham 日志直接相减。负 GT 标签不适合直接送入旧评价脚本当作树实例，因此本轮使用独立、测试覆盖的对齐评价函数。

预设进入下一步的条件：macro Coverage 至少 +0.5 pp，macro F1 不下降，Completeness 下降不超过 0.5 pp，Commission 增加不超过 0.5 pp，至少 3/5 森林 Coverage 不下降。失败也完整保存结果，命令返回 2；不自动启动 attention 或外部测试。

本轮最终应同时报告点归属指标与候选消失/未分配代价。通过本轮后，再考虑补随机种子或进入第 ② 步参数匹配注意力对照。

## 8. 完整结果失败后的事后诊断

本轮五森林 macro Coverage +0.308 pp、F1 -0.595 pp、Commission +1.321 pp，保持 `passed=false`。先排查未分配与竞争改归属的作用，以及计入 FP 的候选状态变化，不直接进入 attention。

更新本分支后，运行新增测试和 CPU 诊断，无需重做缓存、训练或模型推理：

```bash
python -m unittest discover -s tests -p 'test_candidate_mask_mlp.py' -v \
  2>&1 | tee logs/candidate_mask_mlp/unit_tests_assignment_diagnostics.log

python -u tools/mask_refinement/run.py diagnose \
  2>&1 | tee logs/candidate_mask_mlp/diagnose_assignment.log
```

当前测试应发现 28 项（包括后来新增的外部评价检查），测试通过后再运行诊断。`diagnose` 必须使用完整评价，不加 `--pilot` 或 `--smoke`。它先核对保存预测重算的 A0/A1 指标与原 summary，数据不一致则停止。不会修改 `evaluation`、checkpoint 或配置。

输出 `work_dirs/candidate_mask_mlp_a1_s42/diagnostics/assignment/summary.json`，包含原版、完整 A1、仅丢弃点（`abstention_only`）、仅改归属并回填未分配点（`reassignment_only`）的配对比较。每个森林的 `report.json` 和 `candidate_transitions.csv` 解释新增/移除 FP、匹配树丢失、候选消失及树点比例变化。摘要不包含长 GT ID 列表，可直接回传。

报告固定标记 `posthoc=true`、`eligible_for_advancement=false`。它用于解释本次失败，不能据事后对照修改本轮通过判定或声称获得独立验证。

已收到的诊断结果：只丢弃点时 macro F1 -0.794 pp、Coverage +0.393 pp；只改归属并回填未分配点时 F1 -0.570 pp、Coverage -0.120 pp。新增计数 FP 64 个，其中 52 个由原不计入误检的候选转入、12 个由原已匹配候选转入；移除计数 FP 39 个，净增 25。完整分析与下一轮研究假设见结果记录第 6 节。本轮保持失败结论，不再重复运行上述诊断。

## 9. 代码入口

- `tools/mask_refinement/run.py`：完整服务器工作流；
- `tools/mask_refinement/external.py`：已选定 checkpoint 的独立外部准备与配对评价；
- `tree_learn/mask_refinement/data.py`：缓存、候选区域、40D 输入、分层加权采样；
- `tree_learn/mask_refinement/model.py`：条件 MLP 与损失；
- `tree_learn/mask_refinement/inference.py`：无 GT 的全区域分块预测；
- `tree_learn/mask_refinement/evaluation.py`：监督构造、Oracle 与对齐指标；
- `tree_learn/mask_refinement/diagnostics.py`：固定预测的操作对照、点变化与候选评价状态转移；
- `tests/test_candidate_mask_mlp.py`：服务器 CPU 单元与合成闭环测试；
- `tools/pipeline/pipeline.py`：仅新增默认关闭的缓存导出分支，原推理流程保持默认行为。
