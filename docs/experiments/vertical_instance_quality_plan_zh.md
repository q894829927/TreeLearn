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

### 产物

- 所有质量阈值下的 TP、FP、FN、Completeness、Commission 和 F1；
- Oracle F1—Completeness 曲线；
- 最佳阈值仅用于判断上限，不作为最终模型阈值。

### 继续条件

相对 diagnostic r100 baseline，同时满足：

- Oracle detection F1 提升不少于 1.0 个百分点；
- Completeness 下降不超过 1.0 个百分点。

失败则终止整个实例质量评分方向。

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
