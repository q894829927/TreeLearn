# PTv3 阶段开发验证 v1

2026-10-23 用户认为一次跑满预算成本过高，要求在多个节点检查有效性；随后明确批准 **L1W 改为阶段开发验证集**。原“不使用 L1W 早停/选 checkpoint”的协议由此修订，Wytham 继续保留到最终外部评价。阶段选出的模型不能混入原固定 250,000-update final 协议；也不把局部开发指标称为整林成绩。

训练已在第 **2,900** 次更新尝试的完整恢复快照处暂停。快照含 optimizer/scheduler/scaler/RNG/样本游标，SHA256 为 `5a61e228abc542201a4d21414409b2fa209f4aaeb3578f54fd5a6c30f15881ab`；3 次 AMP 跳步仍如实保留，未伪装为 2,900 次实际 optimizer.step。源训练仍冻结在 `649c1749ea654cb42add12db2eacff3da9532c09`，没有修改其模型、训练代码、配置、计划或历史结果。先 SIGSTOP 冻结进程、独立复制并核验完整 checkpoint，再终止进程释放 GPU；暂停记录在运行目录 `stage_validation_v1/pause_record.json`，之后不自动跑满原预算。

固定小区域为 L1W 原坐标 XY `[-24,24] × [-24,24]`，**3,079,275 个 GT 点、24 棵完整标注树**。任何完整 GT 点集越过区域边界的树不参与匹配，其核心内 981,275 个点置为 unknown=-1，绝不改成 non-tree。其他原未知/非树标签保持原义。区域和边界规则在读取模型预测之前冻结，不能根据某个 checkpoint 的效果移动区域。

复用经来源 SHA 核对的旧 L1W 原生完整 tile 缓存：原 0.1m、35m 上下文、8m 内区、stride=0.5、原 Verticality。只选择所有能向核心贡献预测的 **195** 个完整 tile，包含 0.01m 坐标舍入余量；训练输入没有裁剪/限点，选中的推理 tile 也没有截短。tile 输入标签均验证为 unknown，人工 GT 不进入网络或聚类。全量源森林、GT、所有选中 tile 和生成协议 SHA 均记录到 `protocol.json`，每次评价重新校验；不覆盖旧缓存。

预测沿用 main FP32 推理执行（FlashAttention 内部 QKV 精度同上游），不额外降低精度。复用原 `get_pointwise_preds`、8m masks 和 `ensemble`；仅省略保存 backbone features，此项此前已经验证逐点预测不变。对所选内区并集使用 main 固定 Semantic/Verticality/Offset-Z seed、HDBSCAN 和 3D shifted 5NN，最后只在固定核心匹配完整树。**局部聚类的输入范围与整林不同**，所以此分数只能用于同一协议的多节点趋势，不能替代全 L1W benchmark。

每节点保存 detection F1/Precision/Recall、TP/FP/FN、Coverage、Z Extent 分组、Semantic tree recall、Offset XY/XYZ error、逐树诊断、耗时、显存峰值和完整 provenance。未知区域/Commission 和匹配规则复用 main evaluator，不调阈值追分。24 棵树样本较少，个别树的增减即可改变分数；语义召回高本身不代表实例分割有效。

初次对照为相同 seed 42 初始化路径重建的未训练模型（明确不是历史保存的 step-0 快照）与真实 update-2900，再在**同一 ROI/阈值/数据**上运行官方 E2 模型校准区域可分割性。E2 的训练历史与本实验不同，不能据此归因 Backbone。

建议检查节点：

| 节点 | 原计划 epoch | 用途 |
|---|---:|---|
| 2,900 | 第 12 个 epoch 中 | 第一次低成本开发检查 |
| 5,000 | 20 | 第二次趋势检查，仍在 warmup |
| 12,500 | 50 | 原 50-epoch warmup 结束后的判断节点 |
| 25,000 | 100 | 仅在此前有改善、值得继续时再安排 |

不缩短 warmup，不根据单次早期零/低 F1 判定 PTv3 无效。继续的依据是固定区域多次 checkpoint 的实例 F1 与 Coverage 趋势，并用 Offset/语义/seed 诊断解释变化；需要整林确认后才下泛化结论。当前不自动追加后续训练预算。

新评价入口 `tools/experiments/ptv3_stage_v1.py` 提供 prepare/initialize/evaluate，所有输出独立目录。新有界训练入口 `tools/experiments/ptv3_stage_train_v1.py` 可在**不修改冻结源目录**的条件下恢复同一 checkpoint，只将 baseline 原本支持的运行参数 `max_updates` 设为节点上限；配置计算的 `identity.total_updates=250000` 不变，恢复仍执行原模型/optimizer/scheduler/scaler/RNG/游标严格核对，节点完成写 `is_formal_final=false`。

该行政入口的文件和 SHA 单独写入 `bounded_training_manifest.json`，不声称它属于冻结训练的源码 hash。每节点训练使用新目录，保留原训练历史，并把节点终点快照单独留档。原正式入口会对全计划约 473GB 文件重新 SHA 预检，阶段恢复保留此行为，所以 CPU/I/O 启动时间须另计；尚未为了速度更改该验证策略。脚本本身不会排队启动其他节点。运行入口示例：

```bash
env -u PYTORCH_CUDA_ALLOC_CONF OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/szu-stu/miniconda3/envs/TreeLearn/bin/python -u \
  tools/experiments/ptv3_stage_train_v1.py \
  --frozen-source /home/szu-stu/projects/zrx/code/TreeLearn/work_dirs/ptv3_full_formal_v1_20261023_s42/source \
  --root /home/szu-stu/projects/zrx/code/TreeLearn/work_dirs/ptv3_full_formal_v1_20261023_s42/stage_validation_v1 \
  --resume /home/szu-stu/projects/zrx/code/TreeLearn/work_dirs/ptv3_full_formal_v1_20261023_s42/stage_validation_v1/checkpoints/update_2900.pth \
  --stage-end 5000
```

该例仅准备，未据此默认续训。训练在节点自动停下后，再运行同一冻结验证协议。现有每 10 分钟巡检已改为识别主动暂停、区域评价和有界训练，避免把主动暂停报成故障；不会擅自恢复到原 250,000 步。
