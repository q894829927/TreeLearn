# PTv3 完整点云显存优化审计 v1 · 2026-10-23

后续多增强、混合尺寸验证及 allocator OOM 修复见 [短稳定性报告](../ptv3_memory_stability_v1/README_zh.md)。其中保留默认设置和单项设置的失败对照，并记录三个种子各 12 次更新的验证结果；本页继续保留最初固定增强审计的原始数值。

已在 RTX 4090 上完成 **2,677,003 点真实 35m 裁剪**的完整原始输入测试：`block_nested`、micro-batch=1、梯度累积=2，连续 **3 次 forward/backward/AdamW 更新**成功（每次更新含两次完整 forward/backward）。峰值 allocated **18.707 GiB**，reserved **21.164 GiB**。这是一个裁剪、一个固定增强条件下的短测试结果，不代表所有增强、混合尺寸序列或长期训练均已验证。没有启动 500-step Pilot 或正式训练。

分支 `codex/ptv3-full-cloud-memory-v1` 从 `codex/ptv3-treelearn-baseline` 的 `855f831c01089261b93658fa2d9a4507bc43be27` 分出。实现提交为 `8cf62573ef1ebdda0f81ce3289e80e304845a34d`；最终复测的 manifest 记录该提交和干净工作区。TreeLearn main 引用仍为 `fd240ce7caa4c444fe3418aca454dc578bc557d4`，原工作区仍在 `codex/candidate-mask-mlp-step1`。原 PTv3 实验模块、vendor、原配置及 TreeLearn 核心文件逐字节未改，见 [integrity.json](evidence/integrity.json)。新结果全部写入独立目录，没有写入原实验结果、checkpoint 或正式数据池。

**输入与实验边界**

| 输入 | 原始点数 | 原有 voxelizer 输出体素数 | SHA256 |
|---|---:|---:|---|
| `G1N_0.npz`，成功参考裁剪 | 1,060,733 | 866,281 | `71c0d23d44e4825ba9b5f5b533d4b3de8455337cf3ce6c6bbeeb5b4131cd6aae` |
| `G2W_529.npz`，当前池最大真实裁剪 | 2,677,003 | 2,148,502 | `2762a6bcec5cba51ba564833cad2381c88c40049cf7bc00470ae535f047ace3a` |

训练增强种子 `1608637542`，模型/运行种子 42。没有点数上限，没有删点或改变邻域。原 `voxel_size=0.1`、每体素最多 3 点的特征聚合、patch_size=1024、通道、深度、四种序列化顺序、shuffle、所有分支、TreeLearn heads/loss 均保持一致。精度仍为原 CUDA autocast FP16 / Flash FP16；没有额外下调任何算子精度。模型为原 46,167,173 参数版本。

环境：Python 3.10.19、torch 2.0.0 / CUDA 11.8、spconv-cu118 2.3.8、FlashAttention 2.3.6；沿用已有隔离 runtime，未升级共享环境。torch 报告 GPU 总容量为 23.53 GiB。显存表使用 GiB=2^30 bytes，allocated/reserved 是 PyTorch allocator 口径，不包含全部 CUDA 外部分配。

短审计各方案统一使用 AdamW `lr=6.98e-5, weight_decay=0.001`、默认 betas/eps、GradScaler 初始 scale=1024、梯度 clip=1，不执行长期 scheduler。manifest 的 `config` 保留原配置（其中 optimizer.lr=0.003 是长期训练的配置值）；此处实际审计参数以 runner 和 [audit_protocol.json](evidence/audit_protocol.json) 为准。A/B 的 optimizer、loss 和缩放条件相同；这项测试验证完整更新和显存，不检验长期学习率协议或训练收敛。

A/B 每个方案独立进程，复用同一份冻结的完整 voxelizer 输出、v2p 映射、原始 batch、初始权重和 RNG。这样隔离了 GPU voxel-ID 分配顺序的非确定性，未对 voxel 排序或改变输入。两个 fixture 的初始权重文件 SHA 均为 `6c45dd7eda48548ba4e1512e30325f11f3654584dd41660f20a25bd4b92e5df6`。最终最大裁剪测试走原始 NPZ → 原 voxelizer → PTv3 → heads → loss → backward → AdamW，未使用冻结 backbone 输入。

**先行剖析与故障位置**

逐模块前后、每个 ATen 算子均记录同步耗时、allocated/reserved、run peak；模块另有自身执行窗口内的 peak。该窗口包含此前仍存活的张量，不等于该模块独占内存。反向节点使用仅包含字符串的 metadata 标注来源模块，日志不持有输出 Tensor/Point。同步剖析显著影响耗时，性能比较使用后面的独立无剖析运行。

| 原始输入剖析，1 次更新 | 结果 | peak allocated | peak reserved | 剖析墙钟 | Point 创建/释放 |
|---|---|---:|---:|---:|---:|
| 106 万点，原 Attention/MLP checkpoint | 完成更新 | 14.087 | 16.537 | 16.74s | 59/59 |
| 268 万点，原 Attention/MLP checkpoint | forward OOM | 22.219* | 22.934 | 6.27s | 35/35 |
| 268 万点，嵌套 Block checkpoint | 完成更新 | 18.184 | 21.164 | 33.83s | 103/103 |
| 268 万点，纯 Stage checkpoint | backward OOM | 21.556* | 22.547 | 6.46s | 25/25 |

`*` 是失败前已成功分配的峰值，不是该方案完成训练所需峰值。

原最大裁剪失败于 `backbone.dec.dec0.block0.cpe.2` 的 **`aten.native_layer_norm.default`**，申请约 526 MiB 失败；最后成功算子为 `aten._to_copy.default`（autocast 的 FP32 输入转换），最后成功模块为 `...cpe.1`（Linear）。完整 Python 栈和 ATen 失败栈分别保存在 [异常栈](evidence/profile_v4_reference_2677k_exception.txt) 和 [profiles.json](evidence/profiles.json)。峰值已经来自 Decoder 的累计存活激活，不能仅据 OOM 文案推断是 allocator 碎片。

下面列出重点模块的 forward 执行窗口峰值；完整 allocated、reserved、峰值、耗时和重复调用数见三个 [106 万点参考 CSV](evidence/profile_v4_reference_1060k_modules.csv)、[268 万点参考 CSV](evidence/profile_v4_reference_2677k_modules.csv)、[268 万点优化 CSV](evidence/profile_v4_block_nested_2677k_modules.csv)。

| 模块窗口，peak allocated GiB | 106 万参考 | 268 万参考 | 268 万嵌套 Block |
|---|---:|---:|---:|
| Serialization | 1.761 | 4.110 | 4.110 |
| Serialized Pooling enc1 | 3.407 | 8.044 | 4.800 |
| Serialized Pooling enc4 | 6.884 | 14.844 | 6.526 |
| Decoder Unpool dec0 | 10.064 | 21.450 | 9.201 |
| dec0.block0 CPE | 10.588 | 22.219，失败 | 10.690 |
| dec0.block0 Attention | 11.242 | 未到达 | 12.019 |
| dec0.block0 MLP | 11.843 | 未到达 | 12.500 |

优化后全程峰值移至 **`backbone.dec.dec0.block1.mlp.0.fc2` 的 AddmmBackward0 / `aten.sum.dim_IntList`**。纯 Stage 在同一 MLP 的 backward `aten.mm.default` OOM：较大区域在重计算时同时建立多个 Block 的激活，减少边界 Tensor 不一定降低反向峰值。

Point 弱引用计数在成功和异常路径均归零；三步测试清理后的 allocated 在后两步稳定。当前证据支持“单步存活激活及临时工作区过大”，未观察到 Point 对象跨步泄漏。Serialized Pooling 的 CSR/max/mean、cluster、父引用，Unpooling 的 inverse 和 skip 均保留；稀疏卷积 indice cache 也保留以供同级 CPE 复用。

**独立实现及安全边界**

所有实现位于 `tree_learn/experiments/ptv3_memory_v1/`，通过 `build_optimized_model(config, variant)` 显式选择，未修改 vendor。参考实现仍为上游 commit `3229e9b7de1770c8ad17c316f8e349982de509f8` 的原始文件和原 baseline factory。策略清单为 `configs/experiments/ptv3_memory_v1/policy.yaml`。

- `serialization`：仅把输入 Serialization inverse scatter 的 `arange.repeat` 换成零复制 `expand`；encode、argsort、shuffle/RNG 调用顺序不变。避免 4×体素数×8 bytes 的源 Tensor（两个输入分别约 26.4/65.6 MiB），但整体峰值没有下降；不能把它描述为主要收益。Pooling 内的原序列化实现未替换。
- `block`：把 CPE、LayerNorm、残差、Attention、MLP 纳入重计算，替代原内部 Attention/MLP checkpoint。`block_nested` 则保留原内部 checkpoint，再增加 Block 边界，减小重计算期间的临时峰值。
- `stage` / `stage_nested`：只重计算连续的 Blocks，Pool/Unpool 前缀及其 BN 仍执行一次；对非预期的交错 stage 结构明确报错。
- 每次重计算构造全新 Point，只捕获不带梯度的几何/序列化 metadata 和去掉 feature 图的稀疏 Tensor 壳；保留 spconv indices、indice_dict、allocator 等公共 `replace_feature` 语义。闭包不捕获原 Point、pooling_parent 或 feature 图。
- **必须传入两个独立 feature Tensor**：原 Unpooling 更新 `parent.feat`，但 `parent.sparse_conv_feat.features` 仍为投影后的 skip。第一个 Decoder CPE 使用后者。直接统一二者会改变输出，回归测试专门验证了这一陷阱。
- 拒绝重计算区域内的任何 BatchNorm；真实网络的 embedding、pool/unpool、heads BN 均在区域外。使用 torch 2.0 支持的 reentrant checkpoint、`preserve_rng_state=True` 和其 autocast 状态恢复，没有套用新版本才有的 `context_fn`。无可微 feature 输入时回退普通执行，避免 reentrant checkpoint 丢失参数梯度。
- `offload`：原网络加 `save_on_cpu(pin_memory=False)`，原字节与 dtype 保持，反向恢复原设备。spconv 在 ctx 普通属性中持有的 mask/cache 不受 saved-tensor hook 影响；不能声称卸载所有工作区。它是高主存/PCIe 成本的对照及备选方案，非默认路径。

PyTorch 的 [checkpoint 文档](https://docs.pytorch.org/docs/main/utils/checkpoint.html) 解释 RNG/重计算约束，[autograd 文档](https://docs.pytorch.org/docs/stable/autograd.html) 说明 CPU saved-tensor 卸载；本实现的可用接口以**已安装 torch 2.0 源码和实测**为准。

**逐项 A/B 数值检查**

实际比较了全部点的两个预测输出（semantic logits、offset）、459 个未裁剪/已 unscale 的参数梯度、全部 45 个持久 buffer（含 BN running statistics / num_batches_tracked）、backward 后 CPU/NumPy/CUDA RNG。预设接受阈值为输出 `atol=1e-4, rtol=1e-3`、全局梯度相对 L2 ≤1e-3、buffer/RNG 完全一致。原始逐参数误差保存在 [comparisons.json](evidence/comparisons.json) 旁的每个 `comparison_*.json`。这些检查使用一次更新前的相同权重，不把不同更新轨迹当作同权重 A/B。

| 106 万点，逐项相对原实现 | 梯度相对 L2 | 最大绝对梯度差 |
|---|---:|---:|
| 原实现独立重复，非确定性控制 | 0.025842% | 0.00061035 |
| Serialization expand | 0.025938% | 0.00073242 |
| Block | 0.025842% | 0.00061035 |
| Stage | 0.025945% | 0.00061035 |
| 嵌套 Block | 0.025918% | 0.00048828 |
| 嵌套 Stage | 0.026127% | 0.00061035 |
| CPU offload | 0.026066% | 0.00061035 |

所有行的预测最大绝对差和 loss 差均为 **0**，45 个 buffer 与 RNG 均完全一致，梯度有限且通过阈值。固定随机条件并不能消除现有 CUDA scatter/spconv/Flash backward 的归约非确定性；原实现自身重复约 0.02584% 的梯度差与优化差异处于同一量级，因此没有宣称梯度逐位相同，也没有通过更换确定性内核改变基线。

上述阈值是**全局**梯度相对 L2，不代表每个参数都在 0.1% 内。例如 106 万点参考重复中，`enc2.down.proj.bias` 的梯度范数仅约 4.05e-6，逐参数相对差可达 144.65%；嵌套 Block 为 143.84%。这类 BN 前 bias 的近零梯度使相对比值失真。参考重复在范数≥0.001 的参数中最大相对差约 0.824%，嵌套 Block 约 0.756%，对应绝对差在几 e-6 量级。见 [按梯度尺度分析](evidence/gradient_scale_analysis.json) 和完整逐参数表；没有将小梯度的局部差异隐藏在全局指标中，也没有验证长期参数轨迹逐位一致。

268 万点原 GPU 参考无法完成，不能伪造直接 A/B。因此先在 106 万点验证 byte-preserving CPU offload 与参考等价，再以它完成最大输入的同权重数值对照。Block / 嵌套 Block / 嵌套 Stage 的梯度相对 L2 分别为 **0.037172% / 0.036833% / 0.036935%**，最大绝对差分别为 0.00048828 / 0.00048828 / 0.00061035；预测和 loss 差仍为 0，buffer/RNG 完全一致。纯 Stage 在该输入上 OOM，没有给它填通过结果。最大尺寸的该对照为交叉验证证据，不能替代一条不存在的原 GPU 成功运行。

补充的最大尺寸 CPU offload **自身独立重复**得到梯度相对 L2 **0.036838%**、最大绝对差 0.00054932，预测/loss、buffer/RNG 完全一致。这个同尺寸控制的误差与嵌套 Block 的 0.036833% 几乎相同，支持误差主要来自基线已有的非确定性归约，而非改变了网络计算；它仍不是长期训练精度评估。

**显存与训练代价**

106 万点：独立进程各做 3 次完整更新；峰值覆盖三步及 AdamW 状态初始化；耗时取第 2/3 步平均，包含同步、unscale、有限性检查、clip、step，不含 Tensor 导出、剖析和日志落盘。第 1 步约 8.9s（offload 约 14.8s）含稀疏算子首次调优，不能与预热后计时混用。以下均为 physical batch=1、accumulation=1。

| 方案 | peak allocated GiB | peak reserved GiB | 预热后秒/更新 | 相对参考耗时 |
|---|---:|---:|---:|---:|
| 参考 Attention/MLP checkpoint | 14.458 | 16.188 | 0.598 | 1.00× |
| Serialization expand | 14.459 | 16.188 | 0.599 | 1.00× |
| Block | 8.349 | 9.246 | 0.674 | 1.13× |
| Stage | 9.892 | 10.693 | 0.653 | 1.09× |
| 嵌套 Block | 8.220 | 9.143 | 0.737 | 1.23× |
| 嵌套 Stage | 8.595 | 9.658 | 0.721 | 1.20× |
| CPU offload | 3.183 | 4.617 | 6.082 | 10.17× |

嵌套 Block 在这个相同三步口径下比参考减少约 **43.1%** allocated 峰值。这里只验证短运行计时，不提供长期吞吐保证。

268 万点的独立冻结输入、单更新记录如下。初始时尚无 Adam 状态，不能与上表三步峰值直接混算；CPU 主存峰值由 ru_maxrss 测量，offload 在这个输入上约 **32.0 GiB**，其他成功方案约 1.84 GiB。

| 方案 | 结果 | peak allocated GiB | peak reserved GiB |
|---|---|---:|---:|
| 参考 | forward OOM | 21.987* | 22.697 |
| Block | 完成更新 | 18.564 | 21.098 |
| Stage | backward OOM | 21.610* | 22.486 |
| 嵌套 Block | 完成更新 | 18.242 | 20.840 |
| 嵌套 Stage | 完成更新 | 19.422 | 22.268 |
| CPU offload | 完成更新 | 6.554 | 10.635 |

最后原始输入、accumulation=2、各 3 次更新的验证：

| 方案 | 完成 optimizer.step | peak allocated / reserved GiB | 预热后秒/有效更新 | 后两步清理后 allocated GiB |
|---|---:|---:|---:|---:|
| Block | 3/3 | 19.033 / 21.422 | 3.175 | 0.7632 / 0.7632 |
| 嵌套 Block，实现提交后复测 | 3/3 | **18.707 / 21.164** | **3.464** | **0.7587 / 0.7587** |

默认推荐嵌套 Block；希望减少重计算耗时且输入规模有余量时可选普通 Block。最大裁剪不再受当前 OOM 限制，但剩余峰值仍在高分辨率 Decoder MLP backward；更大输入或混合尺寸造成的 allocator 行为尚未验证。optimizer、裁剪输入、精度、网络分支均未替换。

**复现与交付位置**

工作树为 `/home/szu-stu/.codex/worktrees/ptv3-treelearn-baseline/TreeLearn`。所有原始 trace、冻结输入、初始权重、A/B Tensor dump 和日志在独立目录 `/home/szu-stu/projects/zrx/code/TreeLearn/work_dirs/ptv3_memory_v1_20261023`，工作树 `work_dirs/ptv3_memory_v1` 为其链接。大 Tensor/完整 ATen JSONL 不放进 Git；[artifact_inventory.json](evidence/artifact_inventory.json) 提供日志/manifest/trace 的绝对路径、字节数和 SHA，Tensor SHA 则在 fixture manifest 和数值比较 JSON 中。比如最大输入的完整原始失败 trace：[events.jsonl](/home/szu-stu/projects/zrx/code/TreeLearn/work_dirs/ptv3_memory_v1_20261023/profile_v4_reference_2677k/events.jsonl)。

Git 中交付 [全部运行 manifest 与结果](evidence/runs.json)、[最终复测控制台日志](evidence/logs/release_raw_2677k_block_nested_accum2.log)、模块 CSV、逐参数数值比较、完整异常调用栈以及 [7 项回归测试日志](evidence/unit_tests_final.log)。早期 profiler v2 的 ContextVar 未覆盖 autograd worker，峰值汇总不完整，保留原文件但在 runs 中明确排除；修复并补测试后重跑 v3，最终使用附带反向模块来源标注的 v4。报告没有引用 v2 的低估峰值。

下面只启动受限的 3-step 审计，输出目录必须尚不存在：

```bash
cd /home/szu-stu/.codex/worktrees/ptv3-treelearn-baseline/TreeLearn
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/szu-stu/miniconda3/envs/TreeLearn/bin/python \
  tools/experiments/ptv3_memory_v1.py run \
  --variant block_nested \
  --crop data/experiments/ptv3_treelearn/v1/random_crops/npz/G2W_529.npz \
  --steps 3 --accumulation 2 \
  --output work_dirs/ptv3_memory_v1/repro_2677k_block_nested
```

添加 `--profile --operators` 可获得完整剖析，使用另一个新目录。数值 A/B 用 `--fixture work_dirs/ptv3_memory_v1/fixture_1060k --save-tensors --steps 1` 替代 `--crop`，每个 variant 一个新目录，复用同一个 fixture，然后调用 `tree_learn.experiments.ptv3_memory_v1.compare.compare(reference_dir, variant_dir, output_json)`。其他机器先用同一 CLI 的 `prepare --crop ... --output ...` 生成一次 fixture，所有方案共用，不能各自重新 voxelize。入口只接受 1/2/3 steps，不会启动 Pilot。

适配接口示例：

```python
from tree_learn.experiments.ptv3_treelearn.train import load_config
from tree_learn.experiments.ptv3_memory_v1 import build_optimized_model
config = load_config("configs/experiments/ptv3_treelearn/e1_ptv3_micro_checkpoint.yaml")
model = build_optimized_model(config, "block_nested").cuda()
```

后续优先检查 Decoder MLP fc2 backward 的临时张量与归约 workspace，评估拆分重计算边界或融合等价算子，每次仍需单独 A/B。其次优化 Hilbert encode 的大临时工作区，以及 Pooling 的投影后 gather；不改变编码和 CSR 归约顺序。再评估仅选择少量大激活的 byte-preserving CPU offload，以减少当前全量卸载的 PCIe 代价。正式训练前另做多增强种子、混合尺寸与更多 optimizer 更新的短稳定性验证；本轮停在已授权的显存审计范围内。
