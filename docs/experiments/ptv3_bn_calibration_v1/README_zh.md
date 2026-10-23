# PTv3 训练域 BN 校准与冻结 L1W 复测 v1（2026-10-24）

用户在BN/Offset诊断后授权执行独立副本校准及原协议复测。
本轮不恢复训练、不增加update预算、不改变网络或后处理阈值。
BN统计发生有意变更，属于推理统计变体，不是等价代码优化或原实验未经改变的终点模型。

## 预先固定的范围

父快照为5000步 `update_5000_head_fp32_v1.pth`，SHA256：
`c9cc21c42f26f2052e887cdc5b4a9a2ed0f314e22509804846c8cb0d771919fd`。
原快照、optimizer/scheduler/scaler/RNG及采样计划均不改写。
新产物只有net及校准来源，明确 `inference_only=true`、
`resumable_training_checkpoint=false`、`is_formal_final=false`，不能用于冒充完整状态续训。

运行根目录：
`/home/szu-stu/projects/zrx/code/TreeLearn/work_dirs/ptv3_full_formal_v1_20261023_s42`。
本轮输出独立放在 `diagnosis/bn_calibration_v1/`。
校准控制器与统计实现在commit `9dfe126`，依赖原冻结 `source_head_fp32_v1/` 的模型定义与runtime。
原冻结训练、Head FP32和评价源均保持原commit。

固定72个完整训练裁剪：按seed42既有计划前10000个已消费样本，
为18个训练源区域（manifest的source_forest字段）分别选最先出现的4个不同crop，再按区域轮转排列。
采用各自原计划增强seed，完整保留点和实例；不读取L1W/Wytham作统计，
不依赖任何标签或模型预测决定样本。
18/36/72前缀分别覆盖每区域1/2/4个crop。
这是源区域均衡的小型固定校准集合，不等同于对完整训练池的精确统计。

## 统计方法与两项对照

精度固定为原评价的全局FP32和原spconv eval kernel；FlashAttention内部原QKV精度保留。
所有模块始终eval，Dropout/DropPath关闭，不调用backward或optimizer。
所有输入只体素化一次，保存完整voxel和point map；两个版本及所有组共用相同fixture。
不使用每个评价tile自己的batch统计，因此没有评价域自适应或统计泄漏。

统计采用分块FP64中心矩合并，保留跨crop均值变化产生的方差：
`M2 = M2_a + M2_b + (mean_b-mean_a)^2 * n_a*n_b/(n_a+n_b)`。
安装point-weighted总体均值和合并样本的无偏方差到running_mean/running_var。
它与原训练EMA或各crop方差的简单平均不同，需以独立变体解读。
原BN affine参数和num_batches_tracked保持逐位不变；原10000计数只是训练历史，
新校准population的点数另记在逐组报告。

1. **Head-only**：保留全部Backbone统计，仅收集并更新两个Head的BN。
2. **All-sequential**：从原父快照重新开始，按embedding、4组pooling、
   4组decoder unpooling projection、两个Head的顺序，共10组校准。
   每组完整遍历72个crop后安装统计，下游使用已校准上游的eval输出。
   同组的unpooling主/skip投影、两个Head彼此不依赖；实际15组BN执行顺序由GPU hook核验。

每次均执行完整前向，不重计算梯度，不裁剪输入。
Head-only共72次、All-sequential共720次；另有1次完整输入GPU门禁。
每个版本结束核对所有state_dict键、shape/dtype及有限性，
只有所选BN的running_mean/running_var允许改变；其余权重、buffer和counter严格逐位比较。

## 统计稳定性和评价规则

每组记录18/36/72样本的全部通道统计。预先记录36→72的两项提示阈值：
最大均值变化不超过最终标准差0.1倍、最大相对标准差变化不超过0.1。
不满足时标记统计尚不稳定，仍可交付预先固定的诊断复测；
不据此自动扩展校准集或根据L1W结果修改统计。
有限性、完整输入和参数保全检查不通过则停止，不启动评价。

两个校准副本均在任何L1W评价前生成并保存SHA256。
之后串行调用原冻结 `source_validation_v1/tools/experiments/ptv3_stage_v1.py`，
沿用相同config、protocol、195个完整35m输入、48m固定区域和24棵完整树。
评价仅在独立validation root写状态，原 `stage_validation_v1/stage_status.json` 不改写。
核验code/config/protocol/GT哈希与原5000步评价一致，再加入原四项对照。
Wytham不参与任何本阶段统计或选择；不自动恢复训练。

## 入口和证据

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/szu-stu/miniconda3/envs/TreeLearn/bin/python -u \
  tools/experiments/ptv3_bn_calibration_v1.py \
  --run-root /home/szu-stu/projects/zrx/code/TreeLearn/work_dirs/ptv3_full_formal_v1_20261023_s42 \
  --output "$NEW_INDEPENDENT_OUTPUT"
```

输出目录必须不存在。`protocol.json`固定输入选择与方法，
`calibration_inputs.json`记录原crop/增强/fixture哈希及完整输入审计，
逐组JSON记录前缀统计与稳定性，`calibration_result.json`记录参数保全与成本。
原始fixtures和新net保留在独立运行目录，不加入Git。
`evaluation_commands.json`、原评价日志/manifest/metrics与 `comparison.json/.csv`保留完整评价证据。
6项CPU测试覆盖浮点稳定的合并、跨crop方差、参数保护、非有限拒绝和数据流顺序检查。

## 实测结果

两个版本及原四项对照均按同一原评价实现、config、protocol和GT处理。
以下百分比来自完整固定区域，Offset误差单位为m；未把训练loss作为效果判断。

| 模型 / 推理统计版本 | 实例F1 % | Precision % | Recall % | Coverage % | 语义树召回 % | Offset XY / XYZ m |
|---|---:|---:|---:|---:|---:|---:|
| E1初始 | 0 | 0 | 0 | 1.51 | 5.71 | 2.886 / 20.659 |
| E1 u2900原统计 | 0 | 0 | 0 | 15.75 | 20.04 | 2.875 / 16.811 |
| E1 u5000原统计 | 0 | 0 | 0 | 0 | 3.24 | 4.241 / 334.541 |
| u5000 Head-only校准 | 0 | 0 | 0 | 20.87 | 47.56 | 2.864 / 16.617 |
| u5000 All-sequential校准 | 12.90 | 28.57 | 8.33 | 15.93 | 99.34 | 2.837 / 5.120 |
| E2官方成熟模型参考 | 100 | 100 | 100 | 98.24 | 99.96 | 0.313 / 0.390 |

Head-only的TP/FP/FN为0/30/24，All-sequential为2/5/22。
FP按原commission规则统计，不等于区域聚类结果中所有未匹配实例的简单计数。
All-sequential在24棵完整树中匹配2棵，改善有限，尚不能称为可用的单木分割模型。
它的Coverage低于Head-only，说明不能只按Coverage或语义召回评价实例能力。
原生GT对齐距离95分位0.0527m、最大0.1101m；三项5000步评价的坐标和输入特征逐位相同。

| 原始ensemble输出审计 | 原5000 | Head-only | All-sequential |
|---|---:|---:|---:|
| 全部输出点数 | 4,773,224 | 4,773,224 | 4,773,224 |
| Offset Z中位数 m | −353.716 | −4.852 | −18.559 |
| `abs(offset_z)<4`通过点数 | 0 | 1,832,296 | 632,811 |
| Seed点数 | 0 | 49,963 | 3,527 |
| 区域union初始/最终聚类实例 | 0 / 0 | 90 / 90 | 13 / 13 |

输出全部有限。Z Offset的任务是指向基部，并不要求所有点都接近0；
Head-only的Z中位数更接近0、Seed更多，并不意味着误差或实例效果更好。
All-sequential恢复了有限的实例检测，但XY误差仍约2.84m，接近u2900的2.88m，
远高于E2参考的0.313m。其原始X预测范围约[−0.211,0.037]m，Y约[−0.446,0.094]m，
显示当前水平投票位移幅度很小，需要进一步核查是否趋向均值预测及其原因。
目前尚未通过干预证明特征塌缩、训练不足或监督/坐标接口中哪一项是唯一原因。

## 成本、统计稳定性与保全

固定输入总计86,512,115点 / 71,088,014体素；每crop为481,928—2,095,216点。
全部crop来自已消费计划的前135个位置之内，非新的训练样本；点/实例完整保留。
这些crop可能有空间重叠，不能按72个独立森林样本解释统计置信度。

Head-only的36→72最大均值变化0.0221个最终标准差、最大相对标准差变化2.34%。
All-sequential各组对应最大值约0.0459和5.33%，全部满足预设0.1/10%提示阈值。
这仅表示该固定校准集合的前缀统计相对稳定，不保证任意未见区域或未来训练节点都适用。

| 操作 | 耗时秒 | peak allocated / reserved GiB |
|---|---:|---:|
| Head-only校准，72次完整前向 | 38.91 | 7.010 / 18.322 |
| All-sequential校准，720次完整前向 | 351.21 | 7.012 / 18.510 |
| Head-only原协议评价 | 190.31 | 7.193 / 18.637 |
| All-sequential原协议评价 | 194.69 | 7.193 / 18.637 |

控制器总耗时831.57秒，约13.86分钟，包含fixture准备、1次GPU门禁和保存/哈希开销。
没有训练update、反向或optimizer.step。新net不含optimizer/scaler，不能直接用于续训。

Head-only只改变4个running统计张量，All-sequential只改变30个；
全部参数、其他buffer和15个BN计数逐位保留，计数仍为10000。
重新加载序列化后的两个net再次核验，均通过；792次校准前向和390次完整tile评价均成功。
两个新net冻结完成的时间早于第一次L1W评价，未根据评价结果更改样本或统计。
原5000完整快照、原评价输出、原protocol及stage_status保持不变，三个冻结源仍为原commit。
控制器与两个评价进程均已退出，原阶段仍为awaiting_stage_decision，未建立新的长期巡检。

推理副本SHA256：

- Head-only：`ffd7fcb735922105c486166639a416abd7967b5569131730828078468474b85e`
- All-sequential：`6ad6efe34b9b1d7e0baea52e7db13d0b892f081ba3c4ccb3cef76a777c4d148e`

完整small-file证据和原文件SHA256见 [artifact_index.json](evidence/artifact_index.json)，
[comparison.json](evidence/comparison.json) / [CSV](evidence/comparison.csv)及
[completion_verification.json](evidence/completion_verification.json)。
逐组通道统计和运行日志以gzip归档；大体素文件、点级结果及net保留在独立运行目录。

## 结论与后续边界

在不更新任何可训练参数的条件下，重建训练域BN统计明显缓解了数百米偏移，
并使All-sequential获得2个TP。这进一步支持BN失配是原零实例的重要原因。
统计校准并未解决水平Offset预测和完整实例区分的主要差距，不能据此宣布PTv3已有效或优于U-Net。
E2只是成熟参考，并非同初始化/预算的架构公平对照。
当前u5000仍在warmup内，4600后Head精度改变，本次又有显式BN统计变体；
仅24棵完整树的开发区域不足以形成正式性能或跨森林泛化结论。

下一步优先审计XY Offset的预测分布、监督尺度/坐标接口及Backbone特征区分度，
用固定训练样本区分“已经学到但推理失配”和“训练中就只预测均值”，再讨论新的有界训练预算。
本轮没有扩大预算、调整阈值、修改网络或自动将推理副本替换为训练断点。
Wytham继续留到最后。
