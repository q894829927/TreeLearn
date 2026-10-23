# PTv3 5000步 BN / Offset 只读诊断（2026-10-24）

当前零实例的直接原因已定位：原始 Offset Z 整体偏到约 −350m，导致原版
`abs(offset_z) < 4` 的 Seed 条件没有任何点通过。固定输入和权重的对照表明，
Head 的 BN running statistics 与当前激活严重失配，是这一偏移的关键来源。
本次没有恢复训练、更新参数或写回 BN；统计为何发展到这种状态仍需进一步隔离。

## 运行状态与范围

运行根目录：
`/home/szu-stu/projects/zrx/code/TreeLearn/work_dirs/ptv3_full_formal_v1_20261023_s42`。
独立诊断目录为其下 `diagnosis/bn_offset_v1/`。

当前 checkpoint 为 `stage_validation_v1/checkpoints/update_5000_head_fp32_v1.pth`，
SHA256 `c9cc21c42f26f2052e887cdc5b4a9a2ed0f314e22509804846c8cb0d771919fd`。
它是5000步阶段终点，`is_formal_final=false`，不是250000步正式终点。
4600之后双Head采用 `treelearn-both-heads-fp32-v1`；Backbone AMP、block_nested沿用。

400次有界训练尝试已完成：无新的 AMP skip/OOM，全部浮点模型、optimizer和BN状态有限；
15组BN计数均为10000。严格恢复及迁移检查已通过，未发现漏载BN或计数重置。
计数正确和有限性检查通过，不代表推理统计已经适配当前权重。
训练保持 `awaiting_stage_decision`，没有活动训练/评价进程，巡检已在上一阶段完成时删除。

本次只读诊断使用5000步checkpoint的临时模型实例，执行20次成功GPU前向，
不进行反向、optimizer.step、BN校准或checkpoint选择。
三个冻结源目录、原checkpoint、L1W protocol及正式输出均保留。
保全检查见 [preservation_verification.json](evidence/preservation_verification.json)。

## 原始输出与坐标检查

读取原2900和5000的完整 `pointwise_results.npz`，均为4,773,224个ensemble点：

| 检查 | 2900 | 5000 Head FP32 |
|---|---:|---:|
| Offset Z最小/中位/最大，m | −25.748 / −0.466 / −0.333 | −495.808 / −353.716 / −348.088 |
| `abs(z)<4`通过点数 | 3,537,963 | 0 |
| 语义前景点数 | 713,604 | 282,210 |
| 语义前景且verticality通过 | 423,198 | 97,433 |
| 最终Seed点数 | 145,726 | 0 |

两次保存的 `coords` 和 `input_feats` 分别逐元素完全一致，预测全部有限。
原评价的GT对齐距离95分位约0.0527m、最大0.1101m；Z偏移已存在于对齐前的原始预测中。
现有证据排除了数百米坐标平移造成该现象的解释，定位在模型输出和Seed之前。
5000步并非没有任何语义前景，而是所有点均被原Offset Z条件排除。

完整CPU审计及2900/4600/5000的BN向量见 `evidence/saved/result.json.gz`。

## 固定输入下的 BN 因果对照

同一checkpoint、完整输入、体素张量及point map，固定并恢复RNG。
所有模式保持DropPath/Dropout关闭；仅被选中的BN设为
`training=True, track_running_stats=False`，使用当前批次统计且不写running buffers。
每次前向后检查BN完整state_dict逐位未变、RNG相同、输出有限。
因此这些模式是诊断干预，不是直接把整个网络切到train，也不是新的正式评价配方。

训练样本为原故障计划中的完整 `LG1_974.npz`，增强seed=2841047932，
780,587点 / 669,147体素，无输入裁剪或点数上限。
表中Z中位数和Seed为原8m inner区域；GT误差和语义召回使用原训练有效监督mask。
该训练样本GT基部定义与整区域评价GT的参照范围不同，不能直接横比两种误差。
下表采用完整FP32、原稀疏卷积eval执行路径：

| BN模式 | Offset Z中位数 m | Offset XYZ误差 m | 语义树召回 | Seed数 |
|---|---:|---:|---:|---:|
| 全部使用保存统计 | −354.698 | 357.473 | 0.4313 | 0 |
| 仅两个Head使用当前批次统计 | −3.689 | 29.925 | 0.6149 | 3,418 |
| 仅Backbone BN使用当前批次统计 | −518.883 | 481.559 | 1.0000 | 0 |
| 全部BN使用当前批次统计 | −19.243 | 11.599 | 0.9837 | 109 |

Head-only模式与eval模式在全部15个BN入口的逐通道均值、方差完全相同。
因此Head-only的输出变化来自Head BN统计的切换，而非上游激活或随机条件的变化。
仅更换Backbone统计反而恶化Offset；Head和Backbone统计存在耦合，不能单凭某层看似异常就替换。

eval时的最大均值失配按保存标准差归一化：
`max(abs(batch_mean - running_mean) / sqrt(running_var + eps))`：

| BN | 最大失配，保存标准差单位 |
|---|---:|
| `offset_linear.1` | 191.782 |
| `semantic_linear.1` | 96.973 |
| `backbone.dec.dec1.up.proj_skip.1` | 7.497 |
| `backbone.dec.dec0.up.proj.1` | 5.403 |

这是当前批次均值相对保存统计的偏离，不是“单个点达到191σ”的概率陈述。
Offset Head入口的逐通道均值范围约[−3773, 54463]，保存均值约[−3307, 59272]；
大型公共均值的漂移相对保存方差很大，经过BN及后续线性层形成严重推理偏移。

## 两个完整 L1W tile 的复现

同样使用FP32和原稀疏卷积eval路径，不引入区域GT、不做聚类。
中心tile为 `L1W_610.npz`，1,607,827点；东北tile为 `L1W_502.npz`，1,671,942点。
它们均来自冻结195个完整35m tile列表；表中只在原8m inner区统计输出，全部输入点参与前向。

| BN模式 | 中心Z中位数 m | 中心Z条件通过/Seed | 东北Z中位数 m | 东北Z条件通过/Seed |
|---|---:|---:|---:|---:|
| 保存统计 | −353.392 | 0 / 0 | −353.676 | 0 / 0 |
| 仅Head批次统计 | −5.287 | 41,829 / 14 | −6.721 | 30,883 / 26 |
| 仅Backbone批次统计 | −518.891 | 0 / 0 | −511.704 | 0 / 0 |
| 全部BN批次统计 | −20.291 | 13,013 / 65 | −17.653 | 18,240 / 10 |

两个tile都表现出与训练样本相同的大幅统计切换效应。
推理tile的标签为unknown，所以没有计算其GT Offset误差、语义召回或实例F1；
Seed恢复为非零不等于实例分割有效，更不能用这些数替代原固定区域评价。

## FP32 / AMP 独立进程对照

原spconv 2.3.8在纯AMP、稀疏卷积eval模式下不能为该输入找到可用算法，
独立进程仍复现 `can't find suitable algorithm for 0`。
完整异常栈保存在 `evidence/train_amp/exception.txt`；这属于诊断推理kernel限制，
不是原正式训练再次失败，也不能据此声称AMP预测值发生NaN。

为了完成精度对照，仅令SparseConvolutionBase采用其training kernel路径，
其他非BN模块仍eval且无反向，双Head始终FP32。
AMP和FP32在独立进程复用相同体素文件，SHA256：
`ba87604c5540d5936f381570d24b10edab9ce8bcd2458068a45b689c2faa9d99`。
这是共同支持的诊断执行路径，不是正式AMP推理的新实现。

| BN模式 | Backbone AMP的XYZ误差 m | FP32的XYZ误差 m | 绝对差 m |
|---|---:|---:|---:|
| 保存统计 | 357.748 | 357.473 | 0.2755 |
| 仅Head批次统计 | 29.948 | 29.925 | 0.0229 |
| 仅Backbone批次统计 | 481.524 | 481.559 | 0.0345 |
| 全部BN批次统计 | 11.625 | 11.599 | 0.0263 |

另以相同fixture在FP32原eval kernel路径执行完整4模式，表中误差一致。
当前证据表明，在此输入及5000步权重上，瞬时精度切换不足以解释数百米偏移。
它不证明4600之后改变Head训练精度对权重轨迹没有影响；原4617的FP16前向溢出仍是已确认的另一问题。
单次前向耗时含首次kernel初始化差异，仅作为执行日志保留，不作吞吐排名。

## 阶段正式指标仍以原结果为准

以下是此前自动L1W评价的完整固定区域结果，本次诊断没有替换它们。

| 模型 | 实例F1 / P / R | Coverage | 语义树召回 | Offset XY / XYZ m |
|---|---|---:|---:|---:|
| E1初始 | 0 / 0 / 0 | 0.01508 | 0.05710 | 2.886 / 20.659 |
| E1 u2900 | 0 / 0 / 0 | 0.15750 | 0.20039 | 2.875 / 16.811 |
| E1 u5000 Head FP32 | 0 / 0 / 0 | 0 | 0.03243 | 4.241 / 334.541 |
| E2官方参考 | 1 / 1 / 1 | 0.98235 | 0.99955 | 0.313 / 0.390 |

5000阶段400次尝试墙钟合计813.33秒，训练峰值allocated/reserved为17.061/18.861GiB；
自动评价158.92秒，含完整预检的控制器总耗时2172.78秒。
区域仅24棵完整树；u5000仍在50 epochs warmup内，且4600之后Head精度发生变更。
E2是成熟模型参考，不是同初始化/预算的Backbone公平对照；Wytham未参与诊断或阶段选择。
原始对照与阶段核验归档为 `evidence/stage_comparison.json/.csv`、`stage_completion_verification.json`。

## 尚未确定与下一步建议

已确认直接失效链条：保存BN统计失配 → Head输出严重偏移 → Offset Z条件零通过 → 无Seed/无实例。
尚未通过干预确定“BN统计为何失配”的唯一来源。候选因素包括特征和权重快速漂移造成EMA滞后、
训练DropPath与推理的分布差异、不同crop统计，以及物理microbatch=1的统计特性。
本次没有执行物理batch=2对照，不能断言microbatch是唯一原因；15组BN计数正确也不能排除这些因素。

优先建议在独立、版本化的5000步模型副本上，仅使用固定的训练域样本校准BN统计，
不使用L1W/Wytham统计或标签；先验证统计稳定、参数不变，再按原FP32路径和冻结L1W协议重测。
同时记录跨crop的均值/方差和校准范围，比较Head-only与全部BN，明确这是一项推理统计变体。
这些是后续建议，本轮没有执行校准或启动完整区域新评价。
训练样本全部BN批次统计下XYZ误差仍约11.6m，表明修正统计也不保证已学到可用的实例模型。
暂不建议凭训练loss下降继续增加预算，也不以修改阈值、钳制Offset或全网络train模式掩盖问题。

## 复现与证据

入口：[ptv3_bn_offset_diagnosis_v1.py](../../../tools/experiments/ptv3_bn_offset_diagnosis_v1.py)。
要求原TreeLearn环境及隔离runtime，不安装/升级依赖。每次输出目录必须不存在。

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
RUN_ROOT=/home/szu-stu/projects/zrx/code/TreeLearn/work_dirs/ptv3_full_formal_v1_20261023_s42
python tools/experiments/ptv3_bn_offset_diagnosis_v1.py \
  --run-root "$RUN_ROOT" --output "$NEW_TRAIN_DIAG" --case train --precision fp32
python tools/experiments/ptv3_bn_offset_diagnosis_v1.py \
  --run-root "$RUN_ROOT" --output "$NEW_L1W_DIAG" --case l1w_center --precision fp32
```

`--case saved`执行CPU审计；`l1w_ne`选择另一个固定tile。
精度比较分别在新进程使用 `--sparse-kernel training`、`--precision amp/fp32`，
第二次指定第一次的 `--fixture /绝对路径/fixed_voxels.pth`，以保持体素/map完全相同。
使用者需将NEW_*变量设为新的独立目录；无后台启动、恢复训练或写回checkpoint的接口。

`evidence/diagnostic_summary.json`为可读汇总；`artifact_index.json`记录原文件、执行脚本和归档的SHA256。
完整BN通道向量、结果与实际执行脚本以gzip保存，`gzip -cd`可展开。
早期探索脚本修订及未完成的混合精度尝试单列为partial，不作为主结论依据；
后续独立AMP失败进程保存完整异常，5组完整GPU进程的20次成功前向支撑本报告。
大型体素fixture和点级预测仅保留在独立诊断运行目录，不提交到Git。

后续用户已授权并完成训练域独立BN校准及原协议复测，结果见
[训练域BN校准 v1](../ptv3_bn_calibration_v1/README_zh.md)。
该后续变体未更新可训练参数；原始诊断和本报告证据继续保留。
