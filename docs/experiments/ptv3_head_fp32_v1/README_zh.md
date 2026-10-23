# TreeLearn双Head FP32数值安全版本（2026-10-23）

本版本针对第4617次尝试中语义Head首层Linear的FP16前向溢出。两个原始TreeLearn MLP
在局部禁用autocast的FP32区域执行，Backbone继续使用原AMP、FlashAttention和block_nested。
网络结构、参数名、GT、loss、点数、增强、优化器与scheduler配方均保持原定义。
精度执行策略有意改变，版本标识为 `treelearn-both-heads-fp32-v1`；不将它描述为纯等价显存优化。

`enable_head_fp32`只适配实例的Head forward，不包装子模块或改变state_dict键。
BN仍只在正常前向更新一次，不参与重计算。在Head的BN之前检查有限性，避免线性层异常先污染BN。
该接口也可显式用于E0；以后进行架构公平对照时应统一采用同样数值安全策略。

## 验证结果

每项GPU验证使用独立进程，读取相同原始输入、权重及固定RNG；没有写回正式训练状态。
故障对照使用诊断重放的4616状态，最大裁剪使用原始4600快照。

| 验证 | 完整microbatch输入 | 结果 | peak allocated/reserved GiB | 单次冷启动测量秒 |
|---|---|---|---|---|
| 故障旧AMP对照 | 780,587点 + 821,956点的计划 | 第一个microbatch语义loss非有限，未进行反向/更新 | 5.026 / 8.102 | 1.045 |
| 故障Head FP32 | 上述两个完整输入 | 完整forward/backward/AdamW更新成功 | 7.697 / 16.559 | 7.507 |
| 最大裁剪Head FP32 | 2 × 2,677,003点，各2,327,984体素 | 完整forward/backward/AdamW更新成功 | 20.524 / 21.289 | 29.017 |

两个修复测试均检查459组可训练参数梯度全部存在且有限、全部模型/BN浮点状态有限、
15组BN各增加2、optimizer.step未被GradScaler跳过、参数确实更新、无OOM或allocator重试。
所有点和43/57等实例均保留。最大裁剪增强seed为3414486137，沿用原真实35m裁剪及原增强定义。

表中旧AMP对照在前向提前退出，无法与修复后的完整训练步骤直接比较耗时或峰值。
冷启动单次测量包含稀疏卷积kernel初始化、梯度/状态审计，不能代替稳定吞吐测试。
旧AMP基线梯度为不可用值，也不能报告两者有限梯度误差。修复的意义是数值安全，不是零误差复现AMP。
11项CPU测试通过，包括纯FP32输出/输入梯度/参数梯度严格一致、BN状态与RNG、参数身份及state_dict兼容、
BN污染前的异常保护，以及原formal和L1W协议回归。

原诊断的参考实现对照与精度反事实详见 `../ptv3_nonfinite_v1/README_zh.md`。
这些短测不保证未来所有输入或任意长预算都稳定，后续仍需监测。

## 有界恢复与状态迁移

原始4600快照不改写。新版本入口 `ptv3_head_fp32_train_v1.py`：

1. 校验故障/最大裁剪门禁结果、适配器SHA256和原4600快照哈希。
2. 校验原冻结代码及配置身份，建立显式迁移快照。
3. 新identity仅更新code/config哈希以包含新的数值策略；逐项核验model、optimizer、scheduler、
   scaler、全部RNG、采样游标、global_update和epoch与原4600状态一致。
4. 将迁移细节写入 `migration.json`，继续调用原reference.run与原formal显存/BN监测接口。
   原严格恢复检查继续执行，不以关闭身份校验来恢复。
5. 停止于5000（重新执行4601—4616的原计划，再推进到5000；本阶段共400次尝试）。

新冻结源目录为运行根 `source_head_fp32_v1/`。原 `source/`、`source_validation_v1/`及原失败日志保留。
控制器证据目录为 `stage_validation_v1/continuations/to_5000_head_fp32_v1/`，
训练输出为 `stage_validation_v1/training/to_5000_head_fp32_v1/`。
门禁及精度策略见本目录 `evidence/`；实际启动源commit、迁移及实时状态以运行目录JSON为准。

本阶段已于2026-10-23 23:35:53（Asia/Shanghai）启动，冻结源commit为
`e4ea04b3e46eb86bf7d094e6ff72a7b70c7fe50a`，初始控制器/训练PID为3089312/3089314。
迁移8项状态保持检查全部为true，新code/config哈希及父子快照SHA256见
`evidence/startup/migration.json`。23:38的启动观测为正常预检1200/8305；
原native严格恢复检查将在预检后执行，不能将预检期间的completed_updates=0视为训练状态重置。
已为该有界阶段建立10分钟巡检，正常推进保持安静，异常或阶段完成时通知。

预检沿用原native按stop_update=5000核验计划前10000个样本的规则：8305个独立裁剪、
158043461422字节（147.19 GiB），包含已训练前缀。没有修改或缩小输入池。
CPU/GPU预检等待阶段不能按GPU低利用率判故障。

## 自动L1W阶段验证

控制器 `ptv3_head_fp32_controller_v1.py` 仅顺序运行该阶段训练和同一冻结L1W evaluator。
训练达到5000并核验partial-final provenance后，使用原 `source_validation_v1`、原config及protocol
评价版本化的 `update_5000_head_fp32_v1.pth`。原评价本来就是FP32 Head前向，不需要改变推理协议。
新结果与原三项对照保存为 `comparison_at_5000_head_fp32_v1.json/.csv`，明确标识4600起的精度变更。

Wytham不参与当前阶段选择。控制器不会追加到12500或250000，不自动重启失败训练。
阶段完成后停止并等待用户决定；5000快照的 `is_formal_final=false`。

## 5000步阶段完成与BN诊断

2026-10-24已完成400次有界尝试及原协议L1W评价，无新AMP跳步或OOM。
实例F1/P/R/Coverage均为0，语义树召回0.03243，Offset XY/XYZ误差为4.241/334.541m。
15组BN计数均为10000，模型及optimizer浮点状态有限；训练停止，完成巡检已删除。
后续只读对照发现Head BN保存统计与当前激活严重失配，造成Offset Z约−350m的整体偏移，
使原Seed条件零通过。完整证据、精度对照与尚未确定的原因见
[BN / Offset诊断](../ptv3_bn_offset_v1/README_zh.md)。
诊断没有覆盖正式结果、写回BN或恢复训练，原5000快照继续作为最近安全阶段快照保留。

## 可复现入口

```bash
python tools/experiments/ptv3_head_fp32_gate_v1.py \
  --run-root "$RUN_ROOT" --output "$NEW_GATE" --case failure --policy head_fp32
python tools/experiments/ptv3_head_fp32_gate_v1.py \
  --run-root "$RUN_ROOT" --output "$NEW_MAX_GATE" --case maximum --policy head_fp32
python tools/experiments/ptv3_head_fp32_controller_v1.py \
  --run-root "$RUN_ROOT" --attempt "$APPROVED_NEW_ATTEMPT"
```

运行环境沿用TreeLearn conda与隔离FlashAttention runtime；不安装或升级依赖。
控制器要求预先存在且经核验的gate_approval，输出目录不能覆盖已有运行。
