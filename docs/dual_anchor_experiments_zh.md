# 双锚点/树轴形态实验执行与记录

本文档对应 `configs/experiments/` 中的实验配置。目标是在不改变
TreeLearn 骨干网络的情况下，分别验证上部锚点监督、双锚点聚类、
树轴方向损失和轴高形态特征的贡献。

## 1. 配置清单

公共配置：

| 文件 | 用途 |
|---|---|
| `_train_common.yaml` | 三次正式训练共享的优化器、轮数和 DataLoader 设置 |
| `_pipeline_common.yaml` | 五次推理共享的森林路径、保存格式和运行设置 |
| `_evaluation_common.yaml` | 五次评估共享的真值路径、阈值和分区协议 |
| `train_smoke_full.yaml` | 完整方法的两轮冒烟测试 |

正式训练配置：

| 权重 | 配置 | 说明 | 默认输出目录 |
|---|---|---|---|
| T0 | `train_t0_baseline.yaml` | 原版单锚点、L2 offset | `work_dirs/t0_baseline/` |
| T1 | `train_t1_dual_no_axis.yaml` | 双锚点监督，无树轴损失 | `work_dirs/t1_dual_no_axis/` |
| T2 | `train_t2_dual_axis.yaml` | 双锚点监督 + 树轴余弦损失 | `work_dirs/t2_dual_axis/` |

推理与评估配置：

| 实验 | 使用权重 | 上部锚点监督 | 聚类上部 XY | 树轴损失 | 聚类轴高 | 推理配置 | 评估配置 |
|---|---|---:|---:|---:|---:|---|---|
| A | T0 | 否 | 否 | 否 | 否 | `pipeline_a_baseline.yaml` | `evaluate_a_baseline.yaml` |
| B | T1 | 是 | 否 | 否 | 否 | `pipeline_b_upper_aux.yaml` | `evaluate_b_upper_aux.yaml` |
| C | T1 | 是 | 是 | 否 | 否 | `pipeline_c_dual_anchor.yaml` | `evaluate_c_dual_anchor.yaml` |
| D | T2 | 是 | 是 | 是 | 否 | `pipeline_d_dual_axis_loss.yaml` | `evaluate_d_dual_axis_loss.yaml` |
| E | T2 | 是 | 是 | 是 | 是 | `pipeline_e_full.yaml` | `evaluate_e_full.yaml` |

因此 A--E 只需要训练 T0、T1、T2 三个权重。B/C 共用 T1，
D/E 共用 T2。

核心消融的 T0、T1、T2 全部使用原版 L2 offset loss，确保不同组之间
只改变双锚点和树轴相关变量。代码虽然支持 Smooth L1，但它不参与主消融，
避免把损失函数变化误认为双锚点带来的提升。

## 2. 首次运行前修改路径

所有命令均从 TreeLearn 仓库根目录执行。默认配置使用以下相对路径：

```text
data/train/random_crops/npz/
data/val/tiles/npz/
data/model_weights/hais_ckpt_spconv2.pth
data/pipeline/L1W/forest/L1W.laz
data/benchmark/L1W_voxelized01_for_eval.laz
```

如果 AutoDL 数据位于 `/root/autodl-fs/data/`，修改：

1. `configs/_modular/dataset_train.yaml` 中的 `dataset_train.data_root`；
2. `configs/_modular/dataset_test.yaml` 中的 `dataset_test.data_root`；
3. `configs/experiments/_train_common.yaml` 中的 `pretrain`；
4. `configs/experiments/_pipeline_common.yaml` 中的 `forest_path`；
5. `configs/experiments/_evaluation_common.yaml` 中的
   `paths.gt_forest_path`。

推理配置中的 `pretrain` 默认指向第 1000 轮权重。若验证集选择了其他
checkpoint，需要同步修改对应的 pipeline 配置。不能根据测试集指标选择
checkpoint。

## 3. 环境与冒烟测试

```bash
source /root/miniconda3/etc/profile.d/conda.sh
cd /root/autodl-tmp/TreeLearn
conda activate TreeLearn
```

先验证新增训练路径：

```bash
python tools/training/train.py \
  --config configs/experiments/train_smoke_full.yaml \
  --work_dir smoke_full
```

成功标准：

- 两轮训练和验证均结束；
- 生成 `work_dirs/smoke_full/epoch_1.pth` 和 `epoch_2.pth`；
- 日志中存在 `offset_loss`、`upper_offset_loss`、`axis_loss`；
- TensorBoard 中存在 `val/Upper_Offset_Loss` 和
  `val/Axis_Cosine_Error`；
- 所有损失均为有限数值，没有 `NaN` 或显存错误。

冒烟测试权重不参与论文结果。

## 4. 正式训练

依次训练三个模型：

```bash
python tools/training/train.py \
  --config configs/experiments/train_t0_baseline.yaml \
  --work_dir t0_baseline

python tools/training/train.py \
  --config configs/experiments/train_t1_dual_no_axis.yaml \
  --work_dir t1_dual_no_axis

python tools/training/train.py \
  --config configs/experiments/train_t2_dual_axis.yaml \
  --work_dir t2_dual_axis
```

公平性要求：

- 三组使用相同训练 crop、验证 tiles 和预训练初始化；
- 三组统一使用原版 L2 offset loss；
- 不修改 `_train_common.yaml` 中的 epoch、优化器和 batch size；
- 不单独改变某一组的数据增强或训练样本数；
- 记录 AutoDL GPU 型号、总训练时间和峰值显存。

当前代码尚未暴露统一的 `seed` 参数。先进行单次趋势实验即可；正式报告
`mean ± std` 前，应补充 Python、NumPy、PyTorch 和 DataLoader 的统一
随机种子控制。

## 5. 选择 checkpoint

训练过程的验证只提供点级语义、offset 和树轴误差。正式选择 checkpoint
时，应在独立验证森林上运行 pipeline 和官方实例评估。

推荐规则：

1. 优先选择 Detection F1 最高的 checkpoint；
2. F1 接近时选择 Coverage/IoU 更高者；
3. 仍然接近时选择更早的 checkpoint；
4. 选定后修改相应 pipeline 配置中的 `pretrain`；
5. 测试集只在最终配置确定后运行一次。

## 6. 五组推理

按照 A 到 E 的顺序运行：

```bash
python tools/pipeline/pipeline.py \
  --config configs/experiments/pipeline_a_baseline.yaml

python tools/pipeline/pipeline.py \
  --config configs/experiments/pipeline_b_upper_aux.yaml

python tools/pipeline/pipeline.py \
  --config configs/experiments/pipeline_c_dual_anchor.yaml

python tools/pipeline/pipeline.py \
  --config configs/experiments/pipeline_d_dual_axis_loss.yaml

python tools/pipeline/pipeline.py \
  --config configs/experiments/pipeline_e_full.yaml
```

实验 A 默认 `tile_generation: True`，B--E 默认是 `False`，因此同一森林
只生成一次 tiles。如果未先运行 A，需要把首次运行配置的
`tile_generation` 临时设为 `True`。

默认预测结果分别保存在：

```text
data/pipeline/L1W/results_a_baseline/
data/pipeline/L1W/results_b_upper_aux/
data/pipeline/L1W/results_c_dual_anchor/
data/pipeline/L1W/results_d_dual_axis_loss/
data/pipeline/L1W/results_e_full/
```

## 7. 五组官方评估

```bash
python tools/evaluation/evaluate.py \
  --config configs/experiments/evaluate_a_baseline.yaml

python tools/evaluation/evaluate.py \
  --config configs/experiments/evaluate_b_upper_aux.yaml

python tools/evaluation/evaluate.py \
  --config configs/experiments/evaluate_c_dual_anchor.yaml

python tools/evaluation/evaluate.py \
  --config configs/experiments/evaluate_d_dual_axis_loss.yaml

python tools/evaluation/evaluate.py \
  --config configs/experiments/evaluate_e_full.yaml
```

每组评估会在相应预测森林旁创建 `evaluation/`，其中包含：

```text
documentation/evaluate_log.txt
evaluation_results.pt
pred_forest_propagated_to_gt_pointcloud.laz
```

## 8. 结果记录表

实验环境：

| 项目 | 记录 |
|---|---|
| 日期 | |
| Git commit | |
| AutoDL 镜像 | |
| GPU / 显存 | |
| CUDA / PyTorch | |
| 训练数据版本 | |
| 验证森林 | |
| 测试森林 | |

训练记录：

| 权重 | 最终/选中 checkpoint | 最优验证轮次 | 训练时间 | 峰值显存 | 备注 |
|---|---|---:|---:|---:|---|
| T0 baseline | | | | | |
| T1 dual-no-axis | | | | | |
| T2 dual-axis | | | | | |

主消融结果：

| 实验 | F1 ↑ | Omission ↓ | Commission ↓ | Precision ↑ | Recall ↑ | Coverage/IoU ↑ |
|---|---:|---:|---:|---:|---:|---:|
| A Original TreeLearn | | | | | | |
| B Upper-anchor auxiliary | | | | | | |
| C Dual-anchor clustering | | | | | | |
| D Dual-anchor + axis loss | | | | | | |
| E Full method | | | | | | |

相对原版的变化：

| 实验 | ΔF1 | ΔOmission | ΔCommission | ΔPrecision | ΔRecall | ΔCoverage |
|---|---:|---:|---:|---:|---:|---:|
| B - A | | | | | | |
| C - A | | | | | | |
| D - A | | | | | | |
| E - A | | | | | | |

## 9. 低成本参数敏感性

使用已经训练好的 T2 权重，只修改 `pipeline_e_full.yaml` 并使用新的
`results_dir`，无需重新训练。

上部锚点权重：

```text
upper_anchor_weight = 0.5, 1.0, 1.5
axis_height_weight = 0.1
```

轴高权重：

```text
upper_anchor_weight = 1.0
axis_height_weight = 0.0, 0.05, 0.1, 0.2
```

每次只改变一个参数，并为结果使用独立目录。建议先完成 A--E，再根据
完整方法的结果决定是否运行敏感性实验。

## 10. 单次实验完成检查

- [ ] 保存本次实际使用的 train/pipeline/evaluate YAML；
- [ ] 保存训练日志与选中 checkpoint；
- [ ] 保存预测森林和 `evaluation_results.pt`；
- [ ] 填写主消融结果表；
- [ ] 记录失败或异常案例；
- [ ] 不使用测试集挑选 checkpoint 或调整参数；
- [ ] 备份到 `/root/autodl-fs/`，避免 AutoDL 实例释放后丢失。
