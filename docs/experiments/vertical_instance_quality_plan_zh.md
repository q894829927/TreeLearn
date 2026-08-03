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
