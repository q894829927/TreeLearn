# 垂直结构感知实例质量评分：实验顺序与停止规则

## 1. 文档目的

本文件是 `vertical-instance-quality` 分支的唯一实验顺序依据。后续代码、配置、
日志和结果必须按本文顺序产生，不因中途看到的 Wytham 指标临时更改阈值。

研究目标不是简单“加入 Attention”，而是在冻结 TreeLearn 的前提下，利用树体
垂直形态、置信度和投票一致性预测候选实例质量，抑制复杂森林中的假阳性树。

```text
冻结 TreeLearn
      ↓
产生候选树实例
      ↓
8 层归一化高度 token
      ↓
Mean/Max MLP 或单层 Transformer
      ↓
实例有效概率 + 实例 IoU
      ↓
质量阈值过滤
```

理论依据：

- [PointGroup](https://openaccess.thecvf.com/content_CVPR_2020/html/Jiang_PointGroup_Dual-Set_Point_Grouping_for_3D_Instance_Segmentation_CVPR_2020_paper.html) 使用 ScoreNet 评价候选实例；
- [HAIS](https://openaccess.thecvf.com/content/ICCV2021/html/Chen_Hierarchical_Aggregation_for_3D_Instance_Segmentation_ICCV_2021_paper.html) 在聚类后进行噪声过滤和质量评分；
- [SoftGroup](https://openaccess.thecvf.com/content/CVPR2022/html/Vu_SoftGroup_for_3D_Instance_Segmentation_on_Point_Clouds_CVPR_2022_paper.html) 用自顶向下 refinement 抑制假阳性；
- [Mask Scoring R-CNN](https://arxiv.org/abs/1903.00241) 直接回归实例掩码 IoU；
- [SATree](https://doi.org/10.1016/j.ufug.2026.129414) 说明树干、树冠等垂直结构对树实例分割有价值。

## 2. 当前证据与边界

Wytham 实例诊断结果：

| 项目 | 结果 |
|---|---:|
| 有效 TP / FP | 568 / 131 |
| point_density_bbox AUC | 0.8610 |
| confidence_p90 AUC | 0.8525 |
| base_vote_radius_rms AUC | 0.8402 |
| Logistic AUC | 0.9049 |
| 全局 MLP AUC | 0.8969 |

这些结果证明实例质量可预测，但不证明 Attention 有效。全局 MLP 未超过
Logistic，因此 Attention 必须引入原有 35 个全局统计量没有表达的“有序垂直
结构”，并且必须超过不含 Attention 的垂直 MLP。

数据使用边界：

- Wytham 已参与方法选择，只能作为开发集；
- L1W 指标接近饱和，可作回归测试，但不足以单独证明泛化；
- 最终论文必须增加按森林/地块隔离的测试，优先使用 FOR-instance、LAUTx 或
  其他未参与设计的完整标注森林；
- 禁止按预测实例随机拆分训练和测试，必须按森林或空间地块拆分。

## 3. 固定方法定义

### 3.1 候选实例

第一轮使用 r100 候选实例：不按 confidence 删除 seed，不融合 Axis/upper
offset。这样保留较高召回率和足够多的负实例。当前 MLP residual 只负责输出
confidence，不改变 base vote 坐标。

后续必须同时报告：

1. 原始 r100；
2. 当前 seed-ratio r078；
3. r100 + 实例质量评分；
4. r078 + 实例质量评分。

### 3.2 质量标签

对预测实例 `P_i` 和真实树 `G_j`：

```text
quality_i = max_j IoU(P_i, G_j)
positive: quality_i >= 0.50
negative: quality_i < 0.25
ambiguous: 0.25 <= quality_i < 0.50
```

- `quality_i` 用作 IoU 回归目标；
- positive/negative 用于有效性分类；
- ambiguous 不参与 BCE，但仍参与 IoU 回归；
- 边界树、裁切树和无完整 GT 覆盖的实例必须显式标记并排除。

### 3.3 垂直 token

每个候选实例使用自身 `z_min`、`z_max` 归一化高度并划分为 8 层。每层保存：

- 冻结 backbone 32D 特征的 mean 和 max；
- confidence mean/std；
- semantic probability mean；
- verticality mean；
- base-vote XY dispersion；
- 点数占整棵树的比例；
- 固定的归一化高度编码。

每棵树只保存 8 个紧凑 token，不保存完整逐点特征，避免产生超大数据文件。

### 3.4 模型与损失

共享输入和训练设置，只改变聚合器：

```text
Global-MLP：现有 35 个全局统计量
Vertical-MLP：8 层 token → mean/max → MLP
Vertical-Attention：8 层 token → 1 层 Transformer → attention pooling
```

Attention 默认固定为：隐藏维度 64、4 heads、1 层、dropout 0.1。

```text
L = weighted_BCE(validity, label)
    + 0.5 * SmoothL1(predicted_iou, quality_i)
final_score = sigmoid(validity_logit) * predicted_iou
```

原始 TreeLearn、semantic head、offset head和现有 confidence head全部冻结。

## 4. 总体实验顺序

```text
E0 复现与数据审计
 ↓
E1 Oracle IoU 过滤上限
 ↓ PASS
E2 生成训练候选实例数据
 ↓ PASS
E3 Logistic / Global-MLP 基线
 ↓ PASS
E4 Vertical-MLP
 ↓ PASS
E5 Vertical-Attention
 ↓ PASS
E6 集成 pipeline 与阈值锁定
 ↓
E7 外部测试
 ↓
E8 最小消融与论文表格
```

任一阶段未通过停止条件时，不进入下一阶段。

## 5. E0：复现与数据审计

### 操作

1. 固定分支、commit、checkpoint、配置和随机种子；
2. 确认 L1W/Wytham diagnostic 的 prediction、evaluation 和 CSV 对应同一次运行；
3. 列出可用于生成质量标签的完整训练森林及 GT；
4. 检查 GT 是否覆盖非树点、边界树和所有树实例；
5. 记录各森林的点数、树数、扫描平台和点密度。

### 产物

```text
logs/vertical_quality/e0_data_audit.json
logs/vertical_quality/e0_data_audit.md
```

### 继续条件

- 至少有 3 个可按地块隔离的完整标注森林或等价空间分块；
- 训练数据不只包含随机 crop；
- 能够为预测实例计算可靠的最大 IoU。

不满足时先补数据，不训练质量头。

### 实际结果（2026-08-02）

- 状态：**PASS，E0 完成**；
- 审计代码 commit：`175e247ec13c9f20154c6b8d476350dc4abf29cd`；
- 18 个训练森林全部满足完整标注要求，标签覆盖率均为 100%；
- L1W 含 200 棵人工修正树，标签覆盖率 72.814%，满足人工验证集角色要求；
- L1W 与 Wytham 的 prediction、evaluation、instance feature CSV 和 checkpoint 均通过同源性检查；
- Wytham 继续仅作为 development 数据，不进入质量头训练或模型选择；
- FOR-instance 当前不可用，但属于可选外部测试集，不阻塞 E1；
- 所有 Gate 均为 `True`。完整报告位于：

```text
logs/vertical_quality/e0_data_audit.json
logs/vertical_quality/e0_data_audit.md
```

## 6. E1：Oracle IoU 过滤上限

### 目的

用真实最大 IoU 代替网络预测质量，回答“即使质量预测完美，过滤候选实例是否
能提升最终 detection F1”。这是实现网络前最重要的门槛。

### 计划工具

```text
tools/diagnostics/evaluate_instance_quality_oracle.py
```

工具实现后运行形式固定为：

```bash
python tools/diagnostics/evaluate_instance_quality_oracle.py \
  --predictions data/pipeline/wytham/results_instance_diagnostic_mlp_s43_r100/full_forest/wytham_vox0.1.laz \
  --ground_truth data/benchmark/wytham_vox0.1.laz \
  --evaluation data/pipeline/wytham/results_instance_diagnostic_mlp_s43_r100/full_forest/evaluation/evaluation_results.pt \
  --output_dir logs/vertical_quality/e1_oracle_wytham
```

实现后推荐使用固定配置一次运行 L1W sanity 与 Wytham primary gate：

```bash
mkdir -p logs/vertical_quality
set -o pipefail

python -m unittest tests.test_instance_quality_oracle -v \
  2>&1 | tee logs/vertical_quality/e1_oracle_tests.log

python -u tools/diagnostics/evaluate_instance_quality_oracle.py \
  --config configs/experiments/vertical_instance_quality/e1_oracle.yaml \
  2>&1 | tee logs/vertical_quality/e1_oracle_run.log
```

固定扫描阈值为 0.00–1.00、步长 0.05。阈值 0 必须逐项复现已有 evaluation
artifact 的 TP、FP、FN 和预测实例数，否则工具立即停止，不能解释 Oracle 结果。
Oracle 选择仅在 Completeness 下降不超过 1.0 个百分点的候选中最大化 F1；F1
相同时依次选择 Completeness 更高、阈值更低的结果。
GT 与 propagated LAZ 坐标允许最大 0.002 m 差异，用于容纳 `0.001 m` LAS scale
造成的重新量化；实际最大误差写入 JSON。超过该容差仍立即停止。


新增产物：

```text
logs/vertical_quality/e1_oracle_l1w/oracle_results.json
logs/vertical_quality/e1_oracle_l1w/oracle_thresholds.csv
logs/vertical_quality/e1_oracle_l1w/oracle_instance_quality.csv
logs/vertical_quality/e1_oracle_l1w/oracle_f1_completeness.png
logs/vertical_quality/e1_oracle_wytham/oracle_results.json
logs/vertical_quality/e1_oracle_wytham/oracle_thresholds.csv
logs/vertical_quality/e1_oracle_wytham/oracle_instance_quality.csv
logs/vertical_quality/e1_oracle_wytham/oracle_f1_completeness.png
logs/vertical_quality/e1_oracle_summary.md
```

### 产物

- 所有质量阈值下的 TP、FP、FN、Completeness、Commission 和 F1；
- Oracle F1—Completeness 曲线；
- 最佳阈值仅用于判断上限，不作为最终模型阈值。

### 继续条件

相对 diagnostic r100 baseline，同时满足：

- Oracle detection F1 提升不少于 1.0 个百分点；
- Completeness 下降不超过 1.0 个百分点。

失败则终止整个实例质量评分方向。

### 实际结果（2026-08-02）

- 状态：**PASS，E1 完成**；
- L1W sanity：
  - baseline F1：98.422713%；
  - best Oracle F1：100.000000%；
  - F1 提升：1.577287 个百分点；
  - Completeness：100.000000% → 100.000000%；
  - Oracle threshold：0.10；
- Wytham primary：
  - baseline：TP=568、FP=131、FN=309、F1=72.081218%；
  - best Oracle：TP=568、FP=0、FN=309、F1=78.615917%；
  - F1 提升：6.534699 个百分点；
  - Completeness：64.766249% → 64.766249%，下降 0；
  - Oracle threshold：0.50；
- Primary Gate：`True`。

解释边界：Oracle 使用真实 GT 最大 IoU，阈值 0.50 与官方匹配阈值一致，因而
可以理想地去除全部不可匹配预测。它只证明实例质量筛选存在足够上限，不证明该
质量可以从模型特征中学习；E2/E3 必须继续验证可学习性，且不得把 Oracle 阈值
直接当成最终推理阈值。


## 7. E2：生成候选实例训练数据

### 原则

- 使用冻结模型在完整训练森林上产生候选实例；
- grouping 配置与 E1 的 r100 完全相同；
- 先按森林/地块划分 train/validation，再生成模型选择结果；
- Wytham 不进入训练；
- 每个候选实例保存全局统计、8 层 token、IoU、有效标签和来源地块。

### 计划配置与产物

```text
configs/experiments/vertical_instance_quality/gen_quality_proposals.yaml
tools/data_gen/gen_instance_quality_data.py
data/instance_quality/train/*.npz
data/instance_quality/val/*.npz
data/instance_quality/manifest.csv
```

### 继续条件

- 正实例不少于 500；
- 负实例不少于 100；
- train/validation 至少来自 3 个独立地块；
- 任一 `source_plot` 不能同时出现在 train 和 validation；
- 随机抽查至少 50 个实例，IoU 与边界标记正确。

### E2 已实现流程（2026-08-02）

实现文件：

    configs/experiments/vertical_instance_quality/pipeline_quality_template.yaml
    configs/experiments/vertical_instance_quality/gen_quality_proposals.yaml
    tools/data_gen/gen_instance_quality_data.py
    tree_learn/util/instance_quality.py

固定森林级划分：

- train：A1、G1、G2、G3、L2 的 N/W 扫描，以及 LG1、LG2、LG3；
- validation：G4 的 N/W 扫描、L1N、O1 的 N/W 扫描；
- 同一地块的 N/W 扫描永远处于同一 split；
- Wytham、L1W 和后续外部测试集均不进入质量头训练。

每个候选实例保存 35 维全局特征、8 层垂直 token、最大 GT IoU、分类有效掩码、边界掩码和来源地块。分类标签固定为：

    positive: IoU >= 0.50
    negative: IoU < 0.25
    ambiguous: 0.25 <= IoU < 0.50，仅用于 IoU 回归

完整训练森林中，标签 0 表示已标注非树点，仍属于有效监督；只有标签 -1 的未分类点不计入标注覆盖率。这样不会错误丢弃由非树点形成的假阳性候选。

服务器先执行两森林 pilot：

    conda activate TreeLearn
    mkdir -p logs/vertical_quality
    set -o pipefail

    python -m unittest tests.test_instance_quality_data tests.test_instance_quality_generator -v 2>&1 | tee logs/vertical_quality/e2_unit_tests.log

    python - <<'PY'
    from tree_learn.util import get_config
    path = 'configs/experiments/vertical_instance_quality/pipeline_quality_template.yaml'
    cfg = get_config(path)
    print('checkpoint:', cfg.pretrain)
    print('save quality:', cfg.save_cfg.save_quality_training_data)
    print('layers:', cfg.save_cfg.quality_num_layers)
    print('seed ratio:', cfg.grouping.seed_confidence_keep_ratio)
    print('full forest save:', cfg.save_cfg.save_full_forest)
    PY

    nohup python -u tools/data_gen/gen_instance_quality_data.py \
      --config configs/experiments/vertical_instance_quality/gen_quality_proposals.yaml \
      --pilot \
      > logs/vertical_quality/e2_pilot_runner.log 2>&1 < /dev/null &

    echo $! | tee logs/vertical_quality/e2_pilot.pid
    tail -f logs/vertical_quality/e2_pilot_runner.log

pilot 运行期间，逐森林 pipeline 详情位于：

    tail -f data/instance_quality/logs/A1N.log
    tail -f data/instance_quality/logs/G4N.log

pilot 完成后检查：

    cat data/instance_quality/generation_summary.md

只有 gate.passed=True 才执行完整 18 森林生成：

    nohup python -u tools/data_gen/gen_instance_quality_data.py \
      --config configs/experiments/vertical_instance_quality/gen_quality_proposals.yaml \
      > logs/vertical_quality/e2_full_runner.log 2>&1 < /dev/null &

    echo $! | tee logs/vertical_quality/e2_full.pid
    tail -f logs/vertical_quality/e2_full_runner.log

生成器支持断点恢复：已存在且通过结构校验的森林会被跳过。每个森林的最终 NPZ、targets CSV 和 metadata 复制完成后，默认删除该森林的 tiles、体素化点云和临时预测，控制磁盘占用。完整生成后会固定抽取 50 个实例写入 data/instance_quality/manual_audit_sample.csv；在人工抽查完成前，E2 总 gate 保持 False，不得进入 E3。
先用更新后的分层策略重新生成 50 条审计清单；该命令只校验已有 NPZ 并重写 manifest/抽样 CSV，不会重新运行森林 pipeline：

    python -u tools/data_gen/gen_instance_quality_data.py \
      --config configs/experiments/vertical_instance_quality/gen_quality_proposals.yaml \
      2>&1 | tee logs/vertical_quality/e2_refresh_audit_sample.log

对固定 50 个实例执行 artifact 一致性审计：

    python -u tools/diagnostics/audit_instance_quality_artifacts.py \
      --config configs/experiments/vertical_instance_quality/e2_artifact_audit.yaml \
      2>&1 | tee logs/vertical_quality/e2_artifact_audit.log

    cat logs/vertical_quality/e2_artifact_audit/summary.md
    head -n 16 logs/vertical_quality/e2_artifact_audit/audited_instances.csv

该审计逐条核对 CSV/NPZ、正负与模糊 IoU 阈值、边界有效性、来源 split、8 层 occupancy、空层和有限值。它不重新生成已经清理的逐点候选几何；逐点 IoU 算法正确性由 E1 Oracle 对齐检查与 E2 单元测试覆盖。只有审计报告全部为 True，且人工查看 audited_instances.csv 未发现异常后，才显式确认 E2：

    python -u tools/data_gen/gen_instance_quality_data.py \
      --config configs/experiments/vertical_instance_quality/gen_quality_proposals.yaml \
      --manual-audit-confirmed \
      2>&1 | tee logs/vertical_quality/e2_confirm.log

    cat data/instance_quality/generation_summary.md

最终必须显示 manual_audit_confirmed=True 和 passed=True，之后才进入 E3。
### E2 实际结果（2026-08-03）

- 状态：**PASS，E2 完成**；
- 候选实例：9,402；
- 有效实例：9,276；
- 分类有效实例：8,978；
- 正实例：6,179；
- 负实例：2,799；
- train：13 个森林，validation：5 个森林；
- plot/group 泄漏：0；
- 固定 50 实例审计覆盖 14 个地块：positive 22、negative 16、ambiguous 12，并包含边界实例；
- artifact 一致性错误：0；
- manual_audit_confirmed=True，最终 gate=True。

## 8. E3：简单质量评分基线

依次训练并固定三个基线：

1. 最佳单特征阈值；
2. Logistic Regression；
3. Global-MLP。

模型选择只使用 validation。报告 ROC-AUC、FP AP、IoU MAE、Spearman 相关系数
以及阈值过滤后的 detection 指标。

### 继续条件

- Logistic 或 Global-MLP 的 validation ROC-AUC 不低于 0.85；
- FP AP 不低于 0.65；
- 使用预先固定阈值策略后，F1 不低于未过滤 baseline；
- 3 个随机种子均无明显崩溃。

如果 Logistic 最优，仍可进入 Vertical-MLP；但后续模型必须超过 Logistic。

### E3 已实现流程

实现文件：

    configs/experiments/vertical_instance_quality/e3_global_baselines.yaml
    tools/training/train_instance_quality_baselines.py
    tests/test_instance_quality_baselines.py

三组基线严格复用 E2 森林级划分：

1. single_feature：只在 train 上选择方向与最佳单特征；
2. logistic_regression：35 维全局特征、训练集标准化和类别平衡；
3. global_mlp：共享两层 MLP，同时预测有效概率和 IoU，三个随机种子 42/43/44。

Global-MLP 固定为 64/32 hidden、dropout 0.1、AdamW、lr=1e-3、weight decay=1e-3、最多 100 epochs、patience 15。checkpoint 只按 validation IoU MAE、其次 FP AP 选择。候选过滤阈值要求 completeness 下降不超过 1%。

服务器执行：

    python -m unittest tests.test_instance_quality_baselines -v       2>&1 | tee logs/vertical_quality/e3_unit_tests.log

    nohup python -u tools/training/train_instance_quality_baselines.py       --config configs/experiments/vertical_instance_quality/e3_global_baselines.yaml       > logs/vertical_quality/e3_global_baselines.log 2>&1 < /dev/null &

    echo $! | tee logs/vertical_quality/e3_global_baselines.pid
    tail -f logs/vertical_quality/e3_global_baselines.log

结果文件：

    logs/vertical_quality/e3_global_baselines/summary.md
    logs/vertical_quality/e3_global_baselines/summary.json
    logs/vertical_quality/e3_global_baselines/per_seed_metrics.csv
    logs/vertical_quality/e3_global_baselines/validation_predictions.csv
    logs/vertical_quality/e3_global_baselines/checkpoints/global_mlp_seed*.pth

E3 中的 filtered F1 是 validation 候选实例级诊断指标，不冒充完整森林官方 detection F1；完整 pipeline 指标只在 E6 集成后报告。E3 gate 通过后，下一步实现 E4 Vertical-MLP；若 Logistic 最优仍可继续，但 E4/E5 必须超过 Logistic。

### E3 实际结果（2026-08-03）

- 状态：**PASS，E3 完成**；
- 固定对照：Global-MLP；
- Global-MLP ROC-AUC：0.992655，FP AP：0.987779；
- IoU MAE：0.092260，优于 Logistic 的 0.118456；
- Spearman：0.815461，优于 Logistic 的 0.772575；
- 三个种子的 ROC-AUC 范围约 0.0016，无明显崩溃；
- 三个模型的候选实例级最佳阈值均为 0.0，说明 E3 尚未证明实际过滤增益。E4 仍须先证明有序垂直 token 提供额外信息，之后才能实现 Attention。

## 9. E4：Vertical-MLP

### 模型

8 层 token 使用共享 token MLP，然后进行 mean/max pooling，不使用 Attention。
它是 Vertical-Attention 必须击败的直接控制组。

### 训练设置

```text
optimizer: AdamW
lr: 1e-3
weight_decay: 1e-3
epochs: 100
early stopping: 15 epochs
seeds: 42, 43, 44
checkpoint selection: validation IoU MAE，其次 FP AP
```

### 继续条件

- 相对最优全局基线，平均 detection F1 至少不下降；
- IoU MAE 或 FP AP 至少一项明确改善；
- 三个种子的 F1 样本标准差不超过 1.0 个百分点。

失败则不实现 Attention，因为新增垂直结构本身没有提供信息。

### E4 已实现流程

实现文件：

    configs/experiments/vertical_instance_quality/e4_vertical_mlp.yaml
    tools/training/train_vertical_instance_quality.py
    tests/test_vertical_instance_quality.py

模型只使用 E2 的 8 层垂直 token，不拼接 35 维全局特征，也不包含 Attention。每层 token 先经过共享的两层 64 维 MLP，再进行带有效层 mask 的 mean/max pooling，最后通过 64/32 维实例 MLP 输出有效概率和 IoU。标准化参数只由训练森林的有效层计算，空层在标准化后强制保持为 0。

训练参数、损失、随机种子、early stopping 和阈值规则与 E3 一致。E4 直接读取 E3 的固定 `summary.json`，不允许手工抄写或更换对照结果。“明确改善”在运行前固定为：IoU MAE 至少降低 0.002，或 FP AP 至少提高 0.002；同时候选实例级 F1 不下降，三种子 F1 样本标准差不超过 0.01。

服务器执行：

    conda activate TreeLearn
    mkdir -p logs/vertical_quality

    python -m unittest tests.test_vertical_instance_quality tests.test_instance_quality_baselines -v \
      2>&1 | tee logs/vertical_quality/e4_unit_tests.log

    nohup env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
      python -u tools/training/train_vertical_instance_quality.py \
      --config configs/experiments/vertical_instance_quality/e4_vertical_mlp.yaml \
      > logs/vertical_quality/e4_vertical_mlp.log 2>&1 < /dev/null &

    echo $! | tee logs/vertical_quality/e4_vertical_mlp.pid
    tail -f logs/vertical_quality/e4_vertical_mlp.log

结果保存到 `logs/vertical_quality/e4_vertical_mlp/`。脚本只有在全部 E4 Gate 通过后才输出 `PASS: proceed to E5 Vertical-Attention.`；失败时仍保存完整结果，但以非零状态退出并禁止进入 E5。

### E4 实际结果（2026-08-03）

- 状态：**PASS，E4 完成**；
- Vertical-MLP ROC-AUC：0.995627，较 E3 Global-MLP 提高 0.002972；
- FP AP：0.991869，提高 0.004090；
- IoU MAE：0.086063，降低 0.006198；
- 候选实例级 filtered F1：0.976172，三种子样本标准差 0.001295；
- 三个种子均选择阈值 0.01，Completeness 约 99.0%，Commission 约 3.6%–4.1%；
- Spearman 略降 0.005187，但不影响预注册 Gate；
- 所有 E4 Gate 为 True。

该 F1 是固定 validation 候选实例级指标，不能替代 E6 完整 pipeline 和官方匹配评估。E5 必须与 E4 对应种子配对比较，不能只比较最佳单次结果。

## 10. E5：Vertical-Attention

除聚合器外，数据、token、损失、优化器、epoch、随机种子和阈值选择均与
Vertical-MLP 相同。

### 继续条件

相对 Vertical-MLP 三种子均值至少满足一项：

- detection F1 提升不少于 0.5 个百分点；或
- Commission Error 降低不少于 2.0 个百分点，同时 Completeness 下降不超过
  1.0 个百分点。

并且：

- FP AP 不下降；
- 推理额外耗时不超过完整 pipeline 的 10%；
- 至少 2/3 个随机种子优于对应的 Vertical-MLP。

失败时删除论文主方法中的 Attention，保留更简单的最佳质量头。

### E5 已实现流程

实现文件：

    configs/experiments/vertical_instance_quality/e5_vertical_attention.yaml
    tools/training/train_vertical_attention.py
    tests/test_vertical_attention.py

E5 与 E4 共用同一数据加载器、训练集 token 标准化、双任务损失、AdamW、100 epochs、patience 15、随机种子 42/43/44 和阈值规则。共享的两层 Token-MLP 后加入 8 层固定正弦高度编码、隐藏维度 64 的单层 4-head Transformer、128 维 FFN 和 attention pooling；不拼接 35 维全局特征。

E5 直接读取 E4 的 `summary.json` 和 `per_seed_metrics.csv`。继续条件在运行前固定为：平均候选实例 F1 提高至少 0.005，或 Commission 降低至少 0.02 且 Completeness 下降不超过 0.01；FP AP 下降不得超过 0.0001；至少 2/3 个对应种子严格优于 E4。脚本记录参数量和验证实例推理耗时；“完整 pipeline 额外耗时不超过 10%”只能在 E6 集成后正式判定，不以脱离 pipeline 的微基准冒充该结论。

服务器执行：

    conda activate TreeLearn
    mkdir -p logs/vertical_quality

    python -m unittest tests.test_vertical_attention tests.test_vertical_instance_quality -v \
      2>&1 | tee logs/vertical_quality/e5_unit_tests.log

    nohup env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
      python -u tools/training/train_vertical_attention.py \
      --config configs/experiments/vertical_instance_quality/e5_vertical_attention.yaml \
      > logs/vertical_quality/e5_vertical_attention.log 2>&1 < /dev/null &

    echo $! | tee logs/vertical_quality/e5_vertical_attention.pid
    tail -f logs/vertical_quality/e5_vertical_attention.log

结果保存到 `logs/vertical_quality/e5_vertical_attention/`。只有日志出现 `PASS: proceed to E6 quality-filter integration.` 才保留 Attention 并进入 E6；否则以 Vertical-MLP 作为最佳质量头。

## 11. E6：集成 pipeline 与阈值锁定

质量评分发生在完整实例聚类之后：

```text
instance_predictions
      ↓
quality head
      ↓
score < threshold 的实例设为背景/未分类
      ↓
保存 full_forest 与 treewise 结果
```

阈值只能在 validation 上选择一次。选择规则：

1. Completeness 相对 baseline 下降不超过 1.0 个百分点；
2. 在满足第 1 条的阈值中选择 detection F1 最高者；
3. F1 相同时选择更低阈值；
4. 将 checkpoint、阈值和 commit 写入 locked YAML，之后禁止用测试集修改。

必须保留 `quality_filter_enabled: false`，使同一 checkpoint 严格退化为未过滤
baseline。

## 12. E7：正式测试

测试顺序：

1. L1W：回归和饱和场景检查；
2. Wytham：开发结果复核，但明确标记为 development；
3. 完全未参与调参的外部森林/地块：论文主测试结果。

每个数据集报告：

- Completeness、Omission、Commission、Detection F1；
- Segmentation Precision、Recall、Coverage；
- 质量 ROC-AUC、FP AP、IoU MAE、Spearman；
- 参数量、质量评分耗时、总 pipeline 时间和峰值内存。

外部数据优先采用具有正式 development/test 划分的
[FOR-instance](https://arxiv.org/abs/2309.01279)。如传感器域差异过大，必须
将域差异作为实验结论，不能在测试集上重新调阈值。

## 13. E8：最小消融

只在主方法通过 E7 后进行：

| 编号 | 方法 |
|---|---|
| A0 | TreeLearn r100 |
| A1 | seed-ratio r078 |
| A2 | Logistic quality |
| A3 | Global-MLP quality |
| A4 | Vertical-MLP quality |
| A5 | Vertical-Attention quality |
| A6 | A5 去除 IoU 回归，只保留 BCE |
| A7 | A5 去除 vote compactness |

不继续扫描 Transformer 层数、head 数或高度 bin 数。8 bins、1 层、4 heads 是
预先固定设置，避免在小数据集上反复调参。

## 14. 实验记录模板

每次实验在 `logs/vertical_quality/experiment_registry.md` 追加：

```text
实验编号：
日期：
Git commit：
配置文件：
配置 SHA256：
checkpoint：
数据 manifest：
数据 split：
随机种子：
GPU / CUDA / PyTorch：
训练日志：
评估日志：
主要指标：
是否通过 gate：
下一步：
```

禁止覆盖已有 work_dir、日志和 evaluation artifact。所有最终表格必须从保存的
`evaluation_results.pt` 或 JSON 自动生成，不能手工抄写后再计算。

## 15. 总停止规则

出现任一情况立即停止继续增加模块：

- Oracle 上限未通过；
- 无法获得按森林/地块隔离的训练和验证数据；
- Vertical-MLP 未超过或持平全局基线；
- Vertical-Attention 未超过 Vertical-MLP；
- 外部测试 F1 提升低于 0.5 个百分点，且 Commission/Completeness 没有形成明确
  权衡优势；
- 提升只存在于 Wytham，而在独立森林上不能复现。

论文可以保留数据支持的简单方法，但不得以“加入 Attention”本身作为贡献。

## 16. E5 实际结论与 E6 锁定实现（2026-08-03）

### 16.1 E5 实际结果：FAIL

E5 Vertical-Attention 已完整运行，但没有通过预注册 Gate，因此不作为主方法继续：

| 模型 | ROC-AUC | FP AP | IoU MAE | Filtered F1 |
|---|---:|---:|---:|---:|
| E4 Vertical-MLP | 0.995627 | 0.991869 | 0.086063 | 0.976172 |
| E5 Vertical-Attention | 0.995634 | 0.991556 | 0.086244 | 0.853377 |

相对 E4，E5 的 Filtered F1 下降 0.122795，Commission 明显恶化，FP AP 和
IoU MAE 均未改善，且三个配对随机种子中胜出数为 0/3。该结果属于方法失败，
不是程序崩溃。按停止规则：

- 停止继续调整 Transformer 层数、head 数和 dropout；
- 论文主方法锁定为 E4 Vertical-MLP；
- Attention 只可作为负结果或对照，不得包装为有效贡献；
- E6 固定使用 E4 seed 42 checkpoint，因为它在三个 E4 checkpoint 中验证
  IoU MAE 最低（0.085285），且 filtered F1 最高（0.977071）。

### 16.2 E6 固定项

E6 不再重新扫描 Wytham，也不重新选择模型。固定如下：

~~~text
quality checkpoint:
logs/vertical_quality/e4_vertical_mlp/checkpoints/vertical_mlp_seed42.pth

quality score:
sigmoid(validity_logit) * predicted_iou

threshold:
0.01（E4 五个 validation forests 上预先选择）

candidate pipeline:
base-residual MLP seed 43 + r100 base-only 2D clustering
~~~

新增 pipeline 行为：

1. 完成实例聚类和剩余点分配；
2. 从冻结 backbone 特征生成 8 层垂直 token；
3. 严格加载 E4 Vertical-MLP checkpoint；
4. 保存每个实例的 validity、predicted IoU 和 quality score；
5. 当 enabled=True 时将 score < 0.01 的实例设为背景并连续重编号；
6. 当 enabled=False 时只评分，过滤前后标签 SHA256 必须完全相同，否则立即失败；
7. 记录质量评分耗时和完整 pipeline 总耗时。

### 16.3 服务器执行顺序

先更新代码并运行测试：

~~~bash
cd ~/projects/zrx/code/TreeLearn
git switch vertical-instance-quality
git pull --ff-only origin vertical-instance-quality

conda activate TreeLearn
mkdir -p logs/vertical_quality

python -m unittest \
  tests.test_instance_quality_data \
  tests.test_vertical_instance_quality -v \
  2>&1 | tee logs/vertical_quality/e6_unit_tests.log
~~~

确认锁定 checkpoint 和原网络 checkpoint 都存在：

~~~bash
test -f logs/vertical_quality/e4_vertical_mlp/checkpoints/vertical_mlp_seed42.pth
test -f work_dirs/base_residual_mlp_frozen_s43/best_base_residual_xy.pth
~~~

第一阶段只运行 score-only control：

~~~bash
nohup bash -c '
set -e

python -u tools/pipeline/pipeline.py \
  --config configs/experiments/vertical_instance_quality/e6_pipeline_l1w_quality_control.yaml \
  > logs/vertical_quality/e6_pipeline_l1w_quality_control.log 2>&1

python -u tools/evaluation/evaluate.py \
  --config configs/experiments/vertical_instance_quality/e6_evaluate_l1w_quality_control.yaml \
  > logs/vertical_quality/e6_evaluate_l1w_quality_control.log 2>&1
' > logs/vertical_quality/e6_control_runner.log 2>&1 < /dev/null &

echo $! | tee logs/vertical_quality/e6_control.pid
tail -f logs/vertical_quality/e6_control_runner.log
~~~

control 完成后必须先检查：

~~~bash
cat \
  data/pipeline/L1W/results_vertical_quality_e6_control/instance_quality_scores/metadata.json

grep -E \
  "Instance-quality scoring finished|pipeline finished in|Completeness:|Commission Error Rate:|F1 Score:|Coverage:" \
  logs/vertical_quality/e6_pipeline_l1w_quality_control.log \
  logs/vertical_quality/e6_evaluate_l1w_quality_control.log
~~~

只有 metadata 中同时满足以下条件才运行过滤组：

~~~text
filter_enabled = false
score_only_labels_identical = true
checkpoint_seed = 42
threshold = 0.01
~~~

第二阶段运行锁定阈值过滤组：

~~~bash
nohup bash -c '
set -e

python -u tools/pipeline/pipeline.py \
  --config configs/experiments/vertical_instance_quality/e6_pipeline_l1w_quality_filtered.yaml \
  > logs/vertical_quality/e6_pipeline_l1w_quality_filtered.log 2>&1

python -u tools/evaluation/evaluate.py \
  --config configs/experiments/vertical_instance_quality/e6_evaluate_l1w_quality_filtered.yaml \
  > logs/vertical_quality/e6_evaluate_l1w_quality_filtered.log 2>&1
' > logs/vertical_quality/e6_filtered_runner.log 2>&1 < /dev/null &

echo $! | tee logs/vertical_quality/e6_filtered.pid
tail -f logs/vertical_quality/e6_filtered_runner.log
~~~

最后自动汇总并执行 E6 Gate：

~~~bash
python -u tools/diagnostics/summarize_e6_quality_filter.py \
  --config configs/experiments/vertical_instance_quality/e6_summary_l1w.yaml \
  2>&1 | tee logs/vertical_quality/e6_summary_run.log

cat logs/vertical_quality/e6_l1w_summary/summary.md
~~~

E6 只有在以下条件全部满足时才进入 E7：

- score-only 标签摘要完全一致；
- control 与固定 r100 baseline 的精确评估计数和指标一致；
- Completeness 下降不超过 1.0 个百分点；
- F1 不低于 baseline；
- 质量评分耗时不超过完整 pipeline 的 10%；
- checkpoint seed 和阈值与 locked YAML 一致。

如果汇总脚本输出 STOP，不在 Wytham 上改阈值；先定位具体失败 Gate。

## 17. E7 锁定 Wytham 复核

E6 已通过：F1 提升 1.257798 个百分点，Commission 降低 2.468647 个百分点，
Completeness 不下降，质量评分耗时占比 3.777%。因此进入 E7。

Wytham 已参与前期分析，只能标记为 development。E7 固定使用 E4 seed 42
checkpoint 和阈值 0.01，只运行一次，不得根据结果改阈值。

~~~bash
nohup bash -c '
set -e

python -u tools/pipeline/pipeline.py \
  --config configs/experiments/vertical_instance_quality/e7_pipeline_wytham_quality_locked.yaml \
  > logs/vertical_quality/e7_pipeline_wytham_quality_locked.log 2>&1

python -u tools/evaluation/evaluate.py \
  --config configs/experiments/vertical_instance_quality/e7_evaluate_wytham_quality_locked.yaml \
  > logs/vertical_quality/e7_evaluate_wytham_quality_locked.log 2>&1

echo "===== E7 WYTHAM FINISHED $(date) ====="
' > logs/vertical_quality/e7_wytham_runner.log 2>&1 < /dev/null &

echo $! | tee logs/vertical_quality/e7_wytham.pid
tail -f logs/vertical_quality/e7_wytham_runner.log
~~~

完成后执行自动汇总：

~~~bash
python -u tools/diagnostics/summarize_e7_locked_quality.py \
  --config configs/experiments/vertical_instance_quality/e7_summary_wytham.yaml \
  2>&1 | tee logs/vertical_quality/e7_summary_wytham_run.log

cat logs/vertical_quality/e7_wytham_summary/summary.md
~~~

Gate 沿用预注册规则：Completeness 下降不超过 1.0 个百分点，并且 F1 至少提高
0.5 个百分点，或 Commission 至少降低 2.0 个百分点。失败时记录跨域失败，
不允许在 Wytham 上重新调参。通过后下一步不是消融，而是准备从未参与方法选择的
独立外部测试森林。

## 18. E7 Wytham 失败与 E7b 域稳健校准诊断

E7 固定绝对阈值 0.01 在 Wytham 上失败：

| 指标 | Baseline | Filtered | Delta |
|---|---:|---:|---:|
| Completeness | 64.766249% | 57.468643% | -7.297605 pp |
| Commission | 18.741059% | 7.692308% | -11.048751 pp |
| F1 | 72.081218% | 70.836261% | -1.244957 pp |
| Coverage | 57.718038% | 50.777961% | -6.940077 pp |

该结果说明质量分数仍具有误检排序能力，但绝对分值存在明显跨域校准偏移。
阈值删除了 612 / 1862 个候选，误删真实树，因此不能继续使用固定阈值 0.01，
也不能直接在 Wytham 上搜索另一个阈值。

E7b 只使用 E4 固定 validation forests 的预测，跨三个随机种子选择每森林
top-ratio 保留比例。选择规则预先固定为：

1. 三个种子的候选实例 Completeness 均下降不超过 1%；
2. 最大化三个种子的平均 F1；
3. 再最大化最低种子 F1；
4. 仍相同时选择更小的保留比例；
5. Wytham 标签完全不参与比例选择。

服务器运行：

~~~bash
python -u tools/diagnostics/select_quality_keep_ratio.py \
  --config configs/experiments/vertical_instance_quality/e7b_quality_ratio_selection.yaml \
  2>&1 | tee logs/vertical_quality/e7b_keep_ratio_selection_run.log

cat logs/vertical_quality/e7b_keep_ratio_selection/summary.md
~~~

拿到 selected keep ratio 后再实现 pipeline top-ratio 过滤，并在 L1W 做回归。
比例锁定后才允许再运行一次 Wytham development 复核；若再次失败，停止实例质量
过滤路线，不进入消融。

## 19. E7b 实际选择结果与锁定执行（2026-08-03）

E7b 的比例仅由 E4 固定 validation forests、随机种子 42/43/44 选择，未使用
Wytham 标签。预注册的 Completeness 最大下降为 1.00 个百分点。最终锁定：

- keep ratio：`0.85`；
- validation 平均候选 F1：`0.867846`；
- validation 最低 Completeness：`0.993180`；
- `0.85` 是满足 Completeness 约束的可行比例中平均 F1 最高者；
- 后续不得依据 Wytham 结果修改该比例。

过滤采用每个完整森林内的确定性排序：按质量分数降序、实例 ID 升序打破同分，
保留 `ceil(0.85 * N)` 个候选实例。`keep_ratio=1.0` 必须退化为不删除候选实例。

### 19.1 第一阶段：L1W 回归 Gate

先同步代码并确认两个 checkpoint 存在：

~~~bash
conda activate TreeLearn
mkdir -p logs/vertical_quality

test -f logs/vertical_quality/e4_vertical_mlp/checkpoints/vertical_mlp_seed42.pth
test -f work_dirs/base_residual_mlp_frozen_s43/best_base_residual_xy.pth
~~~

运行 L1W pipeline 与官方评估：

~~~bash
nohup bash -c '
set -e

python -u tools/pipeline/pipeline.py \
  --config configs/experiments/vertical_instance_quality/e7b_pipeline_l1w_quality_r085.yaml \
  > logs/vertical_quality/e7b_pipeline_l1w_r085.log 2>&1

python -u tools/evaluation/evaluate.py \
  --config configs/experiments/vertical_instance_quality/e7b_evaluate_l1w_quality_r085.yaml \
  > logs/vertical_quality/e7b_evaluate_l1w_r085.log 2>&1

echo "===== E7b L1W FINISHED $(date) ====="
' > logs/vertical_quality/e7b_l1w_runner.log 2>&1 < /dev/null &

echo $! | tee logs/vertical_quality/e7b_l1w.pid
tail -f logs/vertical_quality/e7b_l1w_runner.log
~~~

运行自动汇总：

~~~bash
python -u tools/diagnostics/summarize_e7b_ratio_quality.py \
  --config configs/experiments/vertical_instance_quality/e7b_summary_l1w.yaml \
  2>&1 | tee logs/vertical_quality/e7b_summary_l1w_run.log

cat logs/vertical_quality/e7b_l1w_summary/summary.md
~~~

L1W Gate 同时检查 checkpoint seed、过滤模式、比例、`ceil(0.85 * N)` 保留数、
Completeness 下降不超过 1 个百分点、F1 不低于固定 baseline、质量评分耗时不超过
完整 pipeline 的 10%。只有输出 `PASS` 才能进入下一阶段。

### 19.2 第二阶段：Wytham 锁定 development 复核

只有 L1W Gate 通过后才执行。本阶段仍使用已经锁定的 checkpoint 和 `0.85`，只运行
一次，不重新选比例：

~~~bash
nohup bash -c '
set -e

python -u tools/pipeline/pipeline.py \
  --config configs/experiments/vertical_instance_quality/e7b_pipeline_wytham_quality_r085_locked.yaml \
  > logs/vertical_quality/e7b_pipeline_wytham_r085_locked.log 2>&1

python -u tools/evaluation/evaluate.py \
  --config configs/experiments/vertical_instance_quality/e7b_evaluate_wytham_quality_r085_locked.yaml \
  > logs/vertical_quality/e7b_evaluate_wytham_r085_locked.log 2>&1

echo "===== E7b WYTHAM FINISHED $(date) ====="
' > logs/vertical_quality/e7b_wytham_runner.log 2>&1 < /dev/null &

echo $! | tee logs/vertical_quality/e7b_wytham.pid
tail -f logs/vertical_quality/e7b_wytham_runner.log
~~~

完成后汇总：

~~~bash
python -u tools/diagnostics/summarize_e7b_ratio_quality.py \
  --config configs/experiments/vertical_instance_quality/e7b_summary_wytham.yaml \
  2>&1 | tee logs/vertical_quality/e7b_summary_wytham_run.log

cat logs/vertical_quality/e7b_wytham_summary/summary.md
~~~

Wytham Gate 沿用预注册条件：Completeness 下降不超过 1.0 个百分点，并且 F1 至少
提高 0.5 个百分点，或 Commission 至少降低 2.0 个百分点；质量评分耗时不超过完整
pipeline 的 10%。若失败，则确认实例质量分数的跨域排序也不足以安全硬过滤，停止硬
删除路线，不在 Wytham 上调整比例；下一步改为保留所有实例、将质量分数用于软加权
或训练域不变的排序目标。若通过，则进入未参与选择的独立外部森林测试，再准备消融。
## 20. E8a 质量引导合并 Oracle

E7b 在 Wytham 上将 568 TP / 131 FP 变为 554 TP / 82 FP：质量排序删除了约
49 个 counted FP，但同时误删约 14 棵真实树。F1 提升 1.150771 个百分点、
Commission 降低 5.847977 个百分点，但 Completeness 下降 1.596351 个百分点，
超过预注册的 1.0 个百分点，因此硬过滤路线正式停止，不能继续在 Wytham 上扫描比例。

E8a 只评估“合并而不删除”的上限。继续使用已经锁定的质量排序和 `keep_ratio=0.85`：

1. 前 85% 为高质量候选，后 15% 为待修复候选；
2. GT 仅在 Oracle 中用于判断两个候选是否属于同一真实树；
3. 低质量候选若存在属于同一 GT 的高质量候选，则将两者点集在 contingency table
   中合并；
4. 找不到安全合并目标的低质量候选保持原样，不删除；
5. Oracle 不产生可部署距离阈值，也不把 Wytham 结果用于调整 `0.85`。

预注册 Gate 同时要求：

- detection F1 至少提高 `0.5` 个百分点；
- Commission 至少降低 `2.0` 个百分点；
- Completeness 下降不超过 `0.5` 个百分点。

L1W 仅作一致性 sanity check；Wytham 仍标记为 development upper bound，不是独立
外部测试。

为保证实例编号、点集分区和质量分数来自同一次运行，Oracle 只接受
`filter_enabled=false` 且 `score_only_labels_identical=true` 的 control artifact。L1W
复用已经验证的 E6 control；Wytham 先生成新的 score-only control：

~~~bash
conda activate TreeLearn
mkdir -p logs/vertical_quality

test -f \
  data/pipeline/L1W/results_vertical_quality_e6_control/instance_quality_scores/metadata.json

nohup bash -c '
set -e

python -u tools/pipeline/pipeline.py \
  --config configs/experiments/vertical_instance_quality/e8a_pipeline_wytham_quality_control.yaml \
  > logs/vertical_quality/e8a_pipeline_wytham_control.log 2>&1

python -u tools/evaluation/evaluate.py \
  --config configs/experiments/vertical_instance_quality/e8a_evaluate_wytham_quality_control.yaml \
  > logs/vertical_quality/e8a_evaluate_wytham_control.log 2>&1

echo "===== E8a CONTROL FINISHED $(date) ====="
' > logs/vertical_quality/e8a_control_runner.log 2>&1 < /dev/null &

echo $! | tee logs/vertical_quality/e8a_control.pid
tail -f logs/vertical_quality/e8a_control_runner.log
~~~

control 完成后先检查 metadata：

~~~bash
cat \
  data/pipeline/wytham/results_vertical_quality_e8a_control/instance_quality_scores/metadata.json
~~~

必须满足 `filter_enabled=false`、`score_only_labels_identical=true`、
`num_instances=1862`，再运行 Oracle：

~~~bash
nohup python -u tools/diagnostics/evaluate_quality_guided_merge_oracle.py \
  --config configs/experiments/vertical_instance_quality/e8a_quality_merge_oracle.yaml \
  > logs/vertical_quality/e8a_merge_oracle_run.log 2>&1 < /dev/null &

echo $! | tee logs/vertical_quality/e8a_merge_oracle.pid
tail -f logs/vertical_quality/e8a_merge_oracle_run.log
~~~

运行中可用以下命令确认进程：

~~~bash
pid=$(cat logs/vertical_quality/e8a_merge_oracle.pid)
ps -p "$pid" -o pid,%cpu,%mem,rss,etime,stat,cmd
~~~

完成后查看：

~~~bash
cat logs/vertical_quality/e8a_merge_oracle_summary/e8a_merge_oracle_summary.md
cat logs/vertical_quality/e8a_merge_oracle_wytham/summary.md
~~~

若 Primary Gate 为 `PASS`，下一步 E8b 不直接写死合并距离，而是重新生成固定 validation
forests 的候选邻接对，保存 base-vote 中心距离、XY 中心距离、树高比、垂直层重叠、
垂直 token 相似度和质量差；先比较单规则、Logistic 与轻量 pair-MLP，再以每森林
Completeness 约束选择模型和阈值。锁定后才允许做 Wytham development 复核。

若 E8a 为 `STOP`，说明即便知道 GT，同树碎片合并也没有足够收益；停止实例质量修复
路线，把 E7b 作为“误检—漏检权衡”消融，下一条主线改为直接提升基础实例分组或语义
召回，而不是继续添加质量头。

## 21. E8a 实际结果与 E8b 邻接对数据阶段（2026-08-04）

E8a Primary Gate 已通过，证明“对低质量碎片进行安全合并”值得继续：

| 数据 | Baseline F1 | Oracle F1 | F1 提升 | Commission 降低 | Completeness 变化 |
|---|---:|---:|---:|---:|---:|
| L1W | 98.422713% | 99.680511% | +1.257798 pp | 2.468647 pp | 0.000000 pp |
| Wytham development upper bound | 72.081218% | 73.123797% | +1.042579 pp | 2.318771 pp | +0.228050 pp |

Wytham 的 Oracle 仅用于确认机制上限，不用于 E8b 的特征、邻居数、候选半径、模型或阈值
选择。E8b 先生成无 Wytham 的邻接对数据。原 data/instance_quality 不覆盖；带绝对几何
信息的新 artifact 单独写到 data/instance_quality_e8b。

### 21.1 E8b 数据定义

对每个森林分别按已锁定的 Vertical-MLP 分数排序，前 85% 是高质量目标候选，后 15%
是待修复源实例。每个源实例在 base-vote XY 中心周围 8 m 的宽松图内最多保留 8 个
最近目标。8 m 和 8 邻居只用于候选召回，不是部署合并规则。

正配对要求源和目标的有效 GT 匹配 ID 相同；其他有效配对为负配对。GT ID、IoU 和
真假树标记只能作为标签或审计字段，后续训练器不得把这些列输入模型。输入特征包括：

- 源/目标质量分数与质量差；
- base-vote 中心距离和原始 XY 中心距离；
- 树高、树高比、Z 范围重叠；
- 有效垂直层重叠和垂直 token cosine；
- occupancy、semantic、verticality、base-vote radius 的垂直剖面差；
- 点数比例。

固定拆分仍为 13 个训练森林和 5 个验证森林，并显式检查森林无跨 split 泄漏。

### 21.2 先运行两森林 pilot

更新代码后，在服务器执行：

~~~bash
cd ~/projects/zrx/code/TreeLearn
git switch vertical-instance-quality
git pull --ff-only origin vertical-instance-quality

conda activate TreeLearn
mkdir -p logs/vertical_quality

nohup python -u tools/data_gen/gen_instance_quality_data.py \
  --config configs/experiments/vertical_instance_quality/e8b_generate_pair_geometry.yaml \
  --pilot \
  > logs/vertical_quality/e8b_geometry_pilot.log 2>&1 < /dev/null &

echo $! | tee logs/vertical_quality/e8b_geometry_pilot.pid
tail -f logs/vertical_quality/e8b_geometry_pilot.log
~~~

几何 pilot 完成后生成 A1N 和 G4N 的邻接对：

~~~bash
set -o pipefail

python -u tools/data_gen/gen_instance_merge_pairs.py \
  --config configs/experiments/vertical_instance_quality/e8b_merge_pair_data.yaml \
  --pilot \
  2>&1 | tee logs/vertical_quality/e8b_pair_pilot.log

cat data/instance_merge_pairs_e8b/pilot_summary.md
~~~

pilot 必须同时满足：存在正配对、存在负配对、所有特征有限、没有 plot split 泄漏。
若 STOP，先查看 positive_source_coverage 和 neighbor_coverage；不得直接训练 E8c。

### 21.3 pilot PASS 后生成完整数据

~~~bash
nohup python -u tools/data_gen/gen_instance_quality_data.py \
  --config configs/experiments/vertical_instance_quality/e8b_generate_pair_geometry.yaml \
  --manual-audit-confirmed \
  > logs/vertical_quality/e8b_geometry_full.log 2>&1 < /dev/null &

echo $! | tee logs/vertical_quality/e8b_geometry_full.pid
tail -f logs/vertical_quality/e8b_geometry_full.log
~~~

生成器会校验并复用已完成的 pilot artifact。若某一个 artifact 不完整，只针对该 plot
使用 --plots PLOT_NAME --force，不要删除整个 output root。

几何完整数据 PASS 后：

~~~bash
set -o pipefail

python -u tools/data_gen/gen_instance_merge_pairs.py \
  --config configs/experiments/vertical_instance_quality/e8b_merge_pair_data.yaml \
  2>&1 | tee logs/vertical_quality/e8b_pair_full.log

cat data/instance_merge_pairs_e8b/full_summary.md
~~~

完整 Gate 要求：

- 训练正配对至少 100；
- 验证正配对至少 30；
- 负配对至少 500；
- 低质量源实例至少 80% 能找到候选邻居；
- 至少 5% 的低质量源实例存在同树正目标；
- 特征全部有限且没有森林拆分泄漏。

### 21.4 E8b PASS 后自动进入 E8c

E8c 固定使用上述 train/validation 拆分，依次比较：

1. 单特征规则（base-vote 距离）；
2. Logistic Regression；
3. 轻量 pair-MLP。

每个源实例只允许选择分数最高的一个目标。模型和合并阈值只在固定 validation forests
上选择。建议预注册 Gate 为：验证 pair precision 不低于 95%，正源实例 recall 不低于
30%，错误合并率不超过 0.5%；pair-MLP 只有在 AP 或约束下 recall 明显优于 Logistic
时才保留。之后先集成并复核 L1W，再锁定一次运行 Wytham development，禁止在 Wytham
结果出来后重新扫描阈值。

如果 E8b 的正目标覆盖率不足 5%，说明当前几何邻接图召回不足。只允许依据训练/验证
森林把候选图扩大一次，再重新审计；若仍不足则停止合并学习路线。如果 E8b PASS 但
E8c 不优于 Logistic，论文方法采用“Vertical-MLP 质量排序 + Logistic 安全合并”，
不为了形式复杂而强行保留神经 pair head。

## 22. E8b 实际结果与 E8c 合并模型比较（2026-08-04）

E8b 完整数据 Gate 已通过：

- 9404 个候选实例，其中 9278 个有效；
- 8156 个邻接对，其中 8027 个有效；
- 1602 个正配对、6425 个负配对；
- 训练正配对 1171，验证正配对 431；
- 邻居覆盖率 98.218%；
- 低质量源实例正目标覆盖率 73.699%；
- 特征全部有限，无 plot split 泄漏。

正目标覆盖率显著高于预注册的 5%，因此候选邻接图不是当前瓶颈，不扩大 8 m 候选半径，
直接进入 E8c。

### 22.1 E8c 固定比较

E8c 只使用 data/instance_merge_pairs_e8b/pairs.csv 中的固定 train/validation 拆分，
不读取 Wytham。依次比较：

1. base-vote 中心距离规则；
2. 标准化特征 Logistic Regression；
3. 两层轻量 pair-MLP。

每个低质量源实例只保留得分最高的一个目标；得分相同时选择较小 target instance ID。
合并阈值只在5个固定 validation forests 上选择，必须同时满足：

- pair precision 不低于 95%；
- 正目标源实例 recall 不低于 30%；
- 错误合并数 / 可评估源实例数不超过 0.5%。

部署 seed 预先锁定为 42，43/44 仅用于稳定性。Logistic 至少比距离规则提高1个百分点
正源 recall 才替换距离规则；pair-MLP 至少比当前简单模型提高2个百分点才会被选中。

### 22.2 服务器运行

先更新代码，然后运行：

~~~bash
cd ~/projects/zrx/code/TreeLearn
git switch vertical-instance-quality
git pull --ff-only origin vertical-instance-quality

conda activate TreeLearn
mkdir -p logs/vertical_quality

nohup env CUBLAS_WORKSPACE_CONFIG=:4096:8 python -u tools/training/train_instance_merge_pairs.py --config configs/experiments/vertical_instance_quality/e8c_pair_merge_models.yaml > logs/vertical_quality/e8c_pair_merge_models_run.log 2>&1 < /dev/null &

echo $! | tee logs/vertical_quality/e8c_pair_merge_models.pid
tail -f logs/vertical_quality/e8c_pair_merge_models_run.log
~~~

判断是否结束：

~~~bash
pid=$(cat logs/vertical_quality/e8c_pair_merge_models.pid)
ps -p "$pid" -o pid,%cpu,%mem,rss,etime,stat,cmd
~~~

完成后查看：

~~~bash
cat logs/vertical_quality/e8c_pair_merge_models/summary.md
cat logs/vertical_quality/e8c_pair_merge_models/per_seed_metrics.csv
~~~

### 22.3 E8c 后续决策

如果 E8c 为 PASS，使用 summary 中的 selected_model、locked seed 42 和
selected_threshold 进入 E8d。E8d 先在 L1W score-only control 上执行真实点标签合并，
必须满足：

- control 标签摘要与 baseline 一致；
- Completeness 下降不超过 0.5 个百分点；
- F1 至少提升 0.5 个百分点或 Commission 至少降低 2.0 个百分点；
- 错误合并后不存在一个源实例同时被合并到多个目标；
- 合并推理耗时不超过完整 pipeline 的 10%。

L1W 通过后才锁定运行一次 Wytham development。不得根据 Wytham 结果重选模型或阈值。

如果 E8c STOP，不实现 pipeline 合并。先检查是否所有模型都因 0.5% 错误合并约束失败；
只允许在固定 validation forests 上把该上限预注册放宽到 1.0% 做一次敏感性分析。
若仍无法达到30%正源 recall，则记录 Oracle 与可学习规则之间的差距，停止该路线。

## 23. E8c 失败与 E8c2 Groupwise 合并头（2026-08-04）

E8c pointwise 三组模型均未通过安全 Gate：

| Model | Pair AUC | Pair AP | Precision | Safe positive-source recall |
|---|---:|---:|---:|---:|
| distance_rule | 0.720212 | 0.446211 | 1.000000 | 2.325581% |
| logistic_regression | 0.783986 | 0.466935 | 1.000000 | 0.387597% |
| pair_mlp | 0.794093 | 0.508841 | 1.000000 | 1.679587% |

E8b 已证明 73.699% 的低质量源实例存在正确候选目标，因此失败原因不是邻居召回不足。
Pointwise BCE 独立判断每个 pair，但部署时实际需要在同一源实例的最多8个候选与“不合并”
之间做一次联合决策；Pair AUC/AP 的改善没有转化为高精度源级召回。

当前0.5%错误合并率在370个验证源上已经允许最多1个错误，但95% precision 仍使召回
低于2.4%，说明主要瓶颈是候选间排序与置信度，不应直接通过放宽错误率进入 pipeline。

### 23.1 E8c2 方法

E8c2 使用共享候选编码器，对每个源实例的候选 target 产生 logits，同时由候选集合的
masked pooling 产生一个 no-merge logit。训练使用 groupwise set likelihood：

- 有正确候选时，最大化所有正确候选概率之和；
- 没有正确候选时，最大化 no-merge 概率；
- padding 候选严格 mask；
- 推理时每个源最多选择一个 target；
- top target 概率作为可校准合并置信度。

仍使用与 E8c 完全相同的17维特征、train/validation forests、95% precision、
30%正源 recall 和0.5%错误合并率。锁定 seed 42，另外要求至少2/3 seeds 通过，
召回标准差不超过0.05，并且平均安全召回至少比 pointwise pair-MLP 提高10个百分点。

### 23.2 服务器运行

~~~bash
cd ~/projects/zrx/code/TreeLearn
git switch vertical-instance-quality
git pull --ff-only origin vertical-instance-quality

conda activate TreeLearn
mkdir -p logs/vertical_quality

nohup env CUBLAS_WORKSPACE_CONFIG=:4096:8 python -u tools/training/train_groupwise_instance_merge.py --config configs/experiments/vertical_instance_quality/e8c2_groupwise_merge.yaml > logs/vertical_quality/e8c2_groupwise_merge_run.log 2>&1 < /dev/null &

echo $! | tee logs/vertical_quality/e8c2_groupwise_merge.pid
tail -f logs/vertical_quality/e8c2_groupwise_merge_run.log
~~~

完成后查看：

~~~bash
cat logs/vertical_quality/e8c2_groupwise_merge/summary.md
cat logs/vertical_quality/e8c2_groupwise_merge/per_seed_metrics.csv
~~~

如果 E8c2 PASS，锁定 seed42 checkpoint 和 summary 中阈值，进入 E8d L1W 真实标签合并
集成。如果 E8c2 STOP，则不继续堆叠注意力或扩大模型；生成固定 validation 的
source-level precision/recall 曲线，执行预注册的一次1.0%错误率敏感性分析。若仍无法
达到30%正源 recall，则停止学习式合并路线，并将 E8a Oracle 作为上限分析而非方法结果。

## 24. E8c2 失败与 E8c3 最终安全敏感性审计（2026-08-04）

E8c2 的实际验证结果为：

- source AP：`0.642634 ± 0.005392`；
- 正源实例 Top-1 correct rate：`0.645995 ± 0.004476`；
- pair precision：`100%`；
- positive-source safe recall：`0.011628 ± 0.003876`；
- unsafe merge rate：`0%`；
- 相对 E8c pointwise safe recall 的增益：`-0.005168`；
- 通过 seed 数：`0/3`。

这说明模型约有 64.6% 的正源实例能够把正确目标排到第一名，但其 top-target 概率无法在
95% precision 约束下区分安全选择和错误选择。候选召回不是主要瓶颈，继续增加注意力层、
隐藏维度或训练轮数没有充分依据。

E8c3 不训练新模型，也不读取 Wytham。它只使用 E8c2 在五个固定 validation forests 上
保存的 `validation_predictions.csv`，生成逐 seed 的完整 source-level 阈值前沿。

### 24.1 唯一决策 Gate

预注册的最后一次放宽为：

- pair precision 不低于 95%；
- unsafe merge rate 从 0.5% 放宽到 1.0%；
- positive-source recall 仍须不低于 30%；
- 锁定 seed 42 必须通过；
- 至少 2/3 seeds 通过；
- 三个 seed 的 recall 样本标准差不超过 0.05。

报告同时给出 90% 和 85% precision、0.5% 和 1.0% unsafe rate 的组合，但低于 95%
precision 的行只用于解释置信度瓶颈，不得用于选择阈值、进入 pipeline 或在 Wytham 上调参。

### 24.2 服务器运行

~~~bash
cd ~/projects/zrx/code/TreeLearn
git switch vertical-instance-quality
git pull --ff-only origin vertical-instance-quality

conda activate TreeLearn
mkdir -p logs/vertical_quality
set -o pipefail

python -u tools/diagnostics/diagnose_groupwise_merge_tradeoff.py \
  --config configs/experiments/vertical_instance_quality/e8c3_groupwise_sensitivity.yaml \
  2>&1 | tee logs/vertical_quality/e8c3_groupwise_sensitivity_run.log

cat logs/vertical_quality/e8c3_groupwise_sensitivity/summary.md
~~~

该诊断只读取 3 × 370 条源级预测，通常数秒内完成。即使 Gate 为 STOP，脚本也会先完整
保存 `frontier.csv`、`sensitivity.csv`、`summary.json` 和 `summary.md`，随后以非零状态退出。

### 24.3 E8c3 后续决策

- 若主 Gate PASS：只锁定 seed 42 在 95% precision、1% unsafe rate 下得到的阈值，进入
  E8d L1W score-only control 和真实合并；L1W 通过后才运行一次 Wytham。
- 若主 Gate STOP：终止学习式实例合并，不再尝试新的 pair/groupwise 网络，也不实现
  E8d。论文中将 E8a 作为 Oracle 上限，把 E8c/E8c2/E8c3 作为可学习安全合并的负结果，
  主线回到 Vertical-MLP 质量排序及域稳健的软使用方式，而不是删除或合并实例。

## 25. E8 路线关闭与 E9 选择性实例分割（2026-08-04）

E8c3 的最终结果为：95% precision、1% unsafe merge rate 下，三个 seed 的安全正源召回仅为
`1.550% / 0.775% / 1.163%`，平均 `1.163%`；即使把 precision 降至 85%，平均召回也仅
`8.140%`。因此学习式实例合并路线正式关闭，不实现 E8d，不再增加 pair/groupwise 网络，
也不得在 Wytham 上重新选择阈值。

E9 将 Vertical-MLP 定位为垂直结构感知的实例可靠性估计器。默认输出仍保留全部实例和原始
TreeLearn 标签；质量分数只用于：

- 给每个预测树输出可靠性；
- 绘制保留率从 50% 到 100% 的 Commission-risk 曲线；
- 绘制相同保留率下的 detection F1 曲线；
- 与 500 次固定随机实例排序进行对照；
- 评价模型是否能把 counted FP 稳定排到低质量端。

E9 使用原始官方评估中固定的 matched TP、counted FP 和 ignored unmatched 身份。它衡量
选择性预测的风险排序，不会在每个保留率重新做 Hungarian matching，也不报告经过删除后的
点级 segmentation precision/recall/coverage。论文中必须把它表述为 selective prediction，
不能表述成默认完整森林输出的精度提升。

### 25.1 固定设置与 Gate

- Vertical-MLP checkpoint 仍锁定为 E4 seed 42；
- 比例网格固定为 `[0.50, 1.00]`，步长 0.01；
- 随机对照固定 500 次，随机种子 `20260804`；
- L1W 使用 E6 score-only control；
- Wytham 使用 E8a score-only control，角色为 final evaluation，无参数选择；
- Commission-area 相对随机排序至少降低 10%；
- F1-area 相对随机排序至少提高 0.5 个百分点；
- 两个数据集均须复现 baseline TP/FP/FN，且 score-only 标签摘要一致。

这些 Gate 只判断质量排序是否形成可发表的风险控制证据，不选择保留率。尤其不得从 Wytham
曲线中挑选最优 ratio 再回到 pipeline 生成新结果。

### 25.2 服务器运行

~~~bash
cd ~/projects/zrx/code/TreeLearn
git switch vertical-instance-quality
git pull --ff-only origin vertical-instance-quality

conda activate TreeLearn
mkdir -p logs/vertical_quality
set -o pipefail

python -u tools/diagnostics/evaluate_selective_instance_quality.py \
  --config configs/experiments/vertical_instance_quality/e9_selective_quality.yaml \
  2>&1 | tee logs/vertical_quality/e9_selective_quality_run.log

cat logs/vertical_quality/e9_selective_summary/summary.md
cat logs/vertical_quality/e9_selective_l1w/summary.md
cat logs/vertical_quality/e9_selective_wytham/summary.md
~~~

主要输出为：

- `score_curve.csv`：Vertical-MLP 排序的逐比例指标；
- `random_curve.csv`：500 次随机对照的均值和标准差；
- `selective_curve.png`：Commission 与 detection F1 双曲线；
- `summary.json` 与 `summary.md`：面积指标和 Gate。

### 25.3 E9 后续决策

- 若 E9 PASS：进入 E10，固定当前所有设置，使用森林级 bootstrap 给 Commission-area reduction
  和 F1-area gain 计算 95% 置信区间，并整理 Vertical-MLP、Global-MLP、随机排序的论文表格。
- 若 E9 STOP：当前 Vertical-MLP 只能作为数据集内质量分类器，缺少跨域风险控制证据；停止
  以实例质量作为论文主创新，转回需要重新训练的分割/聚类机制，不再用后处理筛选包装精度。

## 26. E9 结果与 E10 统计证据（2026-08-04）

E9 已在两个 score-only control 上通过：

| Dataset | Counted FP | Commission-area reduction vs random | F1-area gain vs random |
|---|---:|---:|---:|
| L1W | 5 | 77.855% | +10.962 pp |
| Wytham | 131 | 41.322% | +10.293 pp |

Wytham 未参与 checkpoint、比例网格或 Gate 选择，因此是主要跨域证据。其 85% 保留率固定点为
F1 `73.232%`、Commission `12.893%`、Completeness `63.170%`；完整输出仍保持 100% 实例，
E9 曲线只表示用户主动要求选择性输出时的风险—覆盖率权衡。

L1W 只有 5 个 counted FP，因此 Commission 曲线呈明显阶梯，只作为回归验证，不能单独用于
统计显著性结论。E10 不再训练模型，也不选择 ratio，而是补充两类预注册统计证据。

### 26.1 固定 validation 的配对消融

使用 E3 `global_mlp` 与 E4 `vertical_mlp` 的同一批五个 validation forests、同一组
`[42, 43, 44]` 训练 seed 和相同实例标签。保留比例仍为 0.50 到 1.00、步长 0.01。

- 每个森林内部独立按分数排序；
- invalid/edge 实例参与保留比例计数，但不参与 TP/FP；
- 同时对森林和训练 seed 做 5000 次配对 bootstrap；
- 两个模型在每次 bootstrap 中使用完全相同的森林和 seed 样本；
- 报告 Commission-area reduction 与 F1-area gain 的配对 95% percentile CI；
- 平均差值必须有利于 Vertical-MLP，且至少 2/3 seed 同方向获胜。

置信区间用于描述样本不确定性；由于只有五个验证森林，不把“CI 必须完全高于零”设为 Gate，
但论文必须如实报告 CI 是否跨零。

### 26.2 Wytham 锁定置换检验

在 Wytham 已锁定的 Vertical-MLP 分数和 E9 固定实例身份上执行 10000 次随机排序。每次随机
排序计算完整 Commission-area 和 F1-area，形成随机零假设分布。Vertical-MLP 的单侧 p 值为：

- 随机 Commission-area 不高于观测值的概率；
- 随机 F1-area 不低于观测值的概率。

两个单侧 p 值都必须不超过 0.01。该检验不改变分数、不重新训练、不选择比例，也不重新运行
Hungarian matching。

### 26.3 服务器运行

~~~bash
cd ~/projects/zrx/code/TreeLearn
git switch vertical-instance-quality
git pull --ff-only origin vertical-instance-quality

conda activate TreeLearn
mkdir -p logs/vertical_quality
set -o pipefail

python -u tools/diagnostics/evaluate_quality_statistical_evidence.py \
  --config configs/experiments/vertical_instance_quality/e10_statistical_evidence.yaml \
  2>&1 | tee logs/vertical_quality/e10_statistical_evidence_run.log

cat logs/vertical_quality/e10_statistical_evidence/summary.md
cat logs/vertical_quality/e10_statistical_evidence/validation_ablation.md
cat logs/vertical_quality/e10_statistical_evidence/wytham_permutation.md
~~~

E10 只读取现有小型 CSV、PT 和 score metadata，不处理点云、不占 GPU；预计在 CPU 上数十秒到
数分钟完成。输出还包括 `validation_bootstrap.csv` 和 `external_permutations.csv`，用于论文
复核和绘图。

### 26.4 E10 后续决策

- 若 E10 PASS：进入 E11 论文结果冻结，生成主结果表、Global/Vertical 消融表、统计显著性表、
  两张选择性曲线的论文版图片，以及方法与实验章节初稿；不再增加模型或调参数。
- 若 validation 消融失败但 Wytham 置换通过：可以声称质量分数优于随机排序，但不能声称垂直
  token 优于全局实例特征；需要把方法降级为质量估计框架而非垂直结构创新。
- 若 Wytham 置换失败：停止跨域选择性风险控制主张，E9 只能作为数据集内分析。

## 27. E10 结论与 E10b 最终互补性验证（2026-08-04）

E10 得到的是部分成功，而不是整体失败：

- 在固定 validation 上，Vertical-MLP 相对 Global-MLP 的 Commission-area reduction 为
  `-0.105 pp`，95% CI 为 `[-0.459, +0.315]`；F1-area gain 为 `-0.175 pp`，
  95% CI 为 `[-0.451, +0.137]`。三个训练 seed 均未获胜，因此不能声称垂直 token
  优于全局实例特征。
- 在完全锁定的 Wytham 上，Vertical-MLP 相对随机排序的 Commission-area 与 F1-area
  单侧置换检验均为 `p=0.0001`。因此可以声称该质量分数具有显著的跨域风险排序能力，
  但该结果不能证明增益来自垂直结构。

为区分“全局基线容量不足”和“垂直信息没有互补性”，只允许再进行一次 E10b。E10b 不读取
Wytham，不改变 E2 数据划分、训练 seed、标签、损失、选择性曲线范围或评价规则。

### 27.1 参数匹配模型

- `global_wide`：仅输入 35 维全局实例特征，隐藏层为 `[112, 64]`；
- `global_vertical_fusion`：35 维全局特征投影为 32 维，8×72 垂直 token 经共享 MLP 后做
  masked mean/max pooling，再通过 `[64, 32]` 融合头；
- 两者分别约为 11,394 与 11,842 个参数，差异约 3.8%，满足 10% 参数匹配约束；
- 两个模型都训练 seed `42/43/44`，使用相同 train/validation forests、优化器和早停规则；
- 每个模型同时预测实例有效概率与 IoU，质量分数仍为两者乘积。

### 27.2 严格 Gate

只有以下条件全部满足，才允许锁定 Fusion 并做一次 Wytham score-only 验证：

- 参数量差异不超过 10%；
- Fusion 的平均 Commission-area 与平均 F1-area 均不差于 Global-Wide；
- 两项面积指标都至少在 2/3 个 seed 上获胜；
- 至少一项平均改善达到 0.25 个百分点。

若 Gate 失败，立即关闭“垂直结构增量”路线，不再更换网络、阈值或在 Wytham 上试验。论文只能
把现有模型表述为通用实例质量估计与选择性风险控制框架；若必须保留垂直结构为主创新，则需要
重新设计监督目标或回到分割主干，而不是继续增加后处理分类器。

### 27.3 服务器运行

~~~bash
cd ~/projects/zrx/code/TreeLearn
git switch vertical-instance-quality
git pull --ff-only origin vertical-instance-quality

conda activate TreeLearn
mkdir -p logs/vertical_quality

nohup env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  python -u tools/training/train_global_vertical_fusion.py \
  --config configs/experiments/vertical_instance_quality/e10b_global_vertical_fusion.yaml \
  > logs/vertical_quality/e10b_global_vertical_fusion_run.log 2>&1 < /dev/null &

echo $! | tee logs/vertical_quality/e10b_global_vertical_fusion.pid
tail -f logs/vertical_quality/e10b_global_vertical_fusion_run.log
~~~

完成后查看：

~~~bash
cat logs/vertical_quality/e10b_global_vertical_fusion/summary.md
cat logs/vertical_quality/e10b_global_vertical_fusion/per_seed_metrics.csv
~~~

脚本若以 `RuntimeError: E10b fusion gate failed` 结束，表示实验正常完成但科学 Gate 未通过，
不是程序崩溃。此时不得运行 Wytham。只有摘要明确显示 `passed: True` 时，下一步才是把锁定的
seed 42 Fusion checkpoint 接入 score-only 推理，并与 Global-Wide 在 Wytham 上做一次配对外部验证。

### 27.4 E10b 实际结果与最终决策

E10b 已完成，参数匹配条件成立（Global-Wide 11,394 参数，Fusion 11,842 参数，差异
3.783%），但互补性 Gate 失败：

| Seed | Commission-area reduction | F1-area gain |
|---:|---:|---:|
| 42 | +0.023 pp | -0.034 pp |
| 43 | +0.087 pp | +0.035 pp |
| 44 | -0.004 pp | -0.048 pp |

- 平均 Commission-area reduction：`+0.035 pp`；
- 平均 F1-area gain：`-0.015 pp`；
- Fusion 的 FP-AP、ROC-AUC 与 IoU MAE 较好，但 Spearman 从 `0.822146` 降至 `0.805056`；
- F1-area 只有 1/3 seed 获胜，且改善量远低于预注册的 `0.25 pp`。

因此垂直 token 没有证明对全局实例特征存在稳定、具有部署意义的互补增益。该结论优先于
单点分类指标的轻微改善。正式关闭垂直 MLP、垂直注意力和 Global+Vertical Fusion 路线，
不再运行 Fusion 的 Wytham 实验，也不再通过更换网络或 Gate 寻找正结果。

后续论文路线冻结为“后处理实例质量估计与选择性风险控制”：保留 E9 的风险—覆盖率曲线和
Wytham 随机排序置换证据，但不得把改善归因于垂直结构。若论文必须以注意力或垂直结构为
核心创新，则当前证据不足，需要重新设计分割主干或监督目标，并视为一个全新的研究阶段。