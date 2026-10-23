# PTv3 第4617次尝试非有限值诊断（2026-10-23）

已在独立诊断进程中复现原失败节点。首个非有限模块输出是
`semantic_linear.0`（语义 Head 的第一层 `Linear(64,64)`），输入、权重和偏置均有限。
AMP 将该 Linear 的结果输出为 FP16，第57通道全部780,587个值溢出为正无穷；
后续 BatchNorm 将该通道变成 NaN，最终两个语义 logits 和语义 loss 均为 NaN。
同一次前向的 Offset loss 为有限值10.312067。

本次仅进行故障重放与反事实诊断，未恢复正式训练、启动阶段评价或更改冻结运行配置。
三个可复现入口位于 `tools/experiments/ptv3_nonfinite_*_v1.py`，不修改 vendor 或训练入口。

## 原始状态与范围

- 运行根：`/home/szu-stu/projects/zrx/code/TreeLearn/work_dirs/ptv3_full_formal_v1_20261023_s42`。
- 冻结训练源码：`source/`，commit `649c1749ea654cb42add12db2eacff3da9532c09`。
- 冻结验证源码：`source_validation_v1/`，commit `a32ec3cd6233f37e050ca03d0f8a1bf959324b91`。
- 原训练20:49:33（Asia/Shanghai）因 `FloatingPointError: Nonfinite training loss` 退出。
  最后完整记录为4616，第4617次尝试的第一个 microbatch 失败；未进行该次 optimizer.step。
- 最近安全快照：`stage_validation_v1/training/to_5000/recovery.pth`，update4600，
  SHA256 `e056b26fee0ca643e9bc9067828f6e25703a9993f7d68c735c242bf42e84b8af`。
  哈希、全部浮点状态有限性、15组BN计数9200均已验证。
- 输入 `LG1_974.npz`，augmentation seed `2841047932`，780,587点、43个可见实例，
  669,147个体素。输入未裁点，坐标、特征、Offset GT均有限。
- 原失败时累计 peak allocated/reserved 为18.233/18.934 GiB，异常类型为非有限loss。
  该信息不能推断训练效果。

诊断原始文件保留在运行根 `diagnosis/nonfinite_v1/`，版本化摘要和逐模块完整压缩日志见
本目录 `evidence/`。`evidence/index.json` 记录文件来源及SHA256。

## 一、从4600重放至失败节点

`ptv3_nonfinite_diagnosis_v1.py` 读取冻结训练实现，严格恢复模型、AdamW、scheduler、
GradScaler、全部RNG及采样游标。使用原始35m完整输入、增强、点级加权loss、
microbatch=1/累积2、AMP、梯度反缩放和clip=1；没有将诊断轨迹写回原训练。
数据核验限于诊断实际消费的34个独立裁剪，并校验完整manifest与计划哈希。

重放4601—4616这16次已发生的更新，在4617启用模块前后探针并保存独立的pre-target状态。
32个历史输入摘要全部相同，BN计数均为 `2*update`。在同一个目标microbatch复现非有限loss。
共有615条前向模块输出记录，第一次非有限输出为第606条 `semantic_linear.0`。

| 节点 | 输出精度 | 数值范围或异常 |
|---|---|---|
| decoder末级block0 | FP32 | [-34.583, 26.086]，有限 |
| decoder末级block1 MLP第二Linear | FP16 | [-3312, 2886]，有限 |
| backbone最终特征/语义Head输入 | FP32 | [-3624.939, 3189.105]，有限 |
| semantic_linear.0 | FP16 | 第57通道780,587个 `+Inf`，其他通道有限 |
| semantic_linear.1（BN） | FP16 | 780,587个NaN；其running mean/var也被污染 |
| semantic_linear.3 | FP16 | 1,561,174个NaN |
| semantic loss（50×CE，未加microbatch权重） | FP32 | NaN |
| Offset loss（未加microbatch权重） | FP32 | 10.312067，有限 |

完整模块日志表明到backbone输出为止，观测到的CPE、Attention、MLP和Pool/Unpool模块输出均有限。
最后decoder block的Attention与MLP放大了特征，但仍处于有限范围。
特征各通道均值的范围约[-3621.292,3183.148]，最大通道标准差8.953，存在很大的共同偏置。
这说明当前Linear溢出的直接输入条件；尚不能单凭一次前向确定该放大在长期优化中形成的全部原因。

**重放局限：**恢复时状态完全一致不代表CUDA后续轨迹逐位一致。16次历史更新的语义loss
最大绝对差2.541081，Offset loss最大绝对差1.145990，已有首步语义差0.058409。
体素ID分配、scatter/spconv及FlashAttention反向的非确定性均是潜在来源，未逐一隔离。
诊断保存的4616状态是重放状态，不能冒充原进程丢失的4616快照，也不能作为正式续训起点。

## 二、同一Linear输入和权重的独立复算

`ptv3_nonfinite_operator_probe_v1.py` 读取首次溢出时捕获的完整输入和Linear权重，
禁用TF32，比较AMP、FP32及量化后操作数的FP32运算。

| 运算 | 结果 |
|---|---|
| FP16 AMP Linear，独立重复两次 | 同一第57通道全部溢出，非有限掩码和有限元素均相同 |
| 相同输入/权重的FP32 Linear | 全部有限，整体范围[-57888.590,67188.391] |
| FP32第57通道 | [66826.328,67188.391]，全部超过FP16最大有限值65504 |
| FP16量化操作数、FP32输出 | 全部有限，最大67192.211；输出范围是直接限制 |
| FP32 vs 抽样Float64点积 | 最大绝对误差0.019712 |

同一份Head输入、权重和pre-forward BN状态下，原AMP语义Head输出/CE均为NaN；
诊断用FP32语义Head的logits在[-4.914170,5.003274]，`50×CE=7.415501`，
输入梯度及全部Head参数梯度均有限，BN仅更新一次且running mean/var有限。
监督语义点数54,541；这是单个microbatch的原始语义loss，不能当作完整update loss或验证指标。

FP32只是本次用于定位机制的反事实。它提高局部数值范围，没有被部署为正式训练修复；
本探针未运行backbone反向或optimizer.step，不能据此保证完整续训稳定。
GradScaler作用于loss缩放及反向梯度，当前故障发生在其执行之前，继续降低scale无法直接修复此处前向溢出。

## 三、参考实现与显存适配实现对照

`ptv3_nonfinite_reference_probe_v1.py` 使用同一诊断pre-target状态、一次固定的完整体素化结果
与v2p映射，分别恢复RNG并在training/grad-enabled/AMP模式下运行：

| 实现 | 语义Head输入差异 | 首个语义Linear | 语义loss | Offset loss |
|---|---|---|---|---|
| reference（原Attention/MLP checkpoint） | 对照 | 780,587个Inf | NaN | 10.313528 |
| block_nested | max abs=0，relative L2=0 | 780,587个Inf | NaN | 10.313528 |

两个前向结束后的RNG状态相同，各BN均为9233（本次独立前向只增加一次）。
这表明该冻结状态/输入上的前向溢出同样发生在参考实现，不能仅归因于新增Block重计算。
该对照没有反向或优化器步骤，不证明所有训练轨迹或所有输入均等价。

## 后续处理建议

优先在独立版本中评估两个TreeLearn Head的FP32计算区域，至少覆盖首个Linear及BN，
保留backbone AMP、完整输入、网络结构、监督与后处理。Offset当前首层最大48448，
虽尚有限，也应一并检查其范围。正式采用时需记录精度执行差异，并维持E0/E1共同的数值安全策略。

先使用保存的诊断状态与相同完整样本验证全网络forward/backward、全部梯度和BN有限性、
BN次数与RNG、显存峰值及耗时，再决定是否从原4600步快照恢复到既定5000步终点。
在修复后的首层后增加有限值检查，可在污染BN之前保留更准确的异常位置。
当前原训练保持失败暂停状态，5000步评价未运行，预算未追加。

## 实际执行与复现

实际执行了三个独立GPU诊断进程；三份入口均通过 `py_compile`。
完整重放记录、原始异常、三个结果文件和运行源码副本均保存在独立目录。
原训练/验证源码仍为原commit且工作树干净，原4600快照SHA256不变。

以下命令的输出路径必须不存在。原始环境为
`/home/szu-stu/miniconda3/envs/TreeLearn/bin/python`，设置
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`。
在优化分支的仓库目录使用下列入口；示例中的 `RUN_ROOT` 是上面的运行根路径，
`REPLAY` 是第一次命令新建的诊断输出。

```bash
python tools/experiments/ptv3_nonfinite_diagnosis_v1.py \
  --run-root "$RUN_ROOT" --output "$REPLAY"
python tools/experiments/ptv3_nonfinite_operator_probe_v1.py \
  --run-root "$RUN_ROOT" --replay "$REPLAY" --output "$OPERATOR_PROBE"
python tools/experiments/ptv3_nonfinite_reference_probe_v1.py \
  --run-root "$RUN_ROOT" --replay "$REPLAY" --output "$REFERENCE_PROBE"
```

三个入口均使用冻结 `source/` 的训练、数据和模型代码，保留原始源码作对照。
探针不支持把诊断状态自动接回正式训练，也不会启动L1W或Wytham评价。
