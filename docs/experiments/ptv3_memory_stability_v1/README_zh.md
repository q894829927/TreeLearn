# PTv3 多增强与混合尺寸短稳定性验证 · 2026-10-23

下一步已完成：在上一轮 `block_nested` 等价重计算基础上，修复了**增强变大、尺寸切换和 spconv 外部工作区引起的 allocator OOM**。最终三个种子 42/43/44 各完成 **12/12 次 AdamW 更新**，共 72 次完整 forward/backward；micro-batch=1、累积=2。另将此前失败的最难增强从冷启动连续更新 3 次，峰值 allocated **21.204 GiB**、reserved **21.732 GiB**，在同一 RTX 4090 上成功。

本轮为受限短测试，没有启动 500-step Pilot 或长期训练。仍在 `codex/ptv3-full-cloud-memory-v1`，运行实现提交为 `06bde21`。原始 [固定增强审计](../ptv3_memory_v1/README_zh.md) 保持原数值；这轮结果扩大了已验证条件，也记录了原默认 allocator 的新失败条件。

**发现与修复依据**

同一个 2,677,003 点裁剪，增强种子 `3414486137` 产生 **2,327,984 个体素**，比原固定增强的 2,148,502 个体素多约 8.35%。这来自原数据增强，不是调整 voxel_size。默认 allocator 在混合尺寸第 7 次更新的第二个 micro-batch 失败，重复运行在同一位置再次失败。

异常捕获时的 native allocator 快照显示约 **3.322 GiB inactive split**，最大可见空闲块仅 **0.894 GiB**，当时新申请约 **1.11 GiB**。这些统计支持不可释放碎片导致分配失败，而非 Point 跨步泄漏。完整 [默认设置异常栈](evidence/stability_20261023_v3/runs/default/s42/exception.txt)、[memory snapshot](evidence/stability_20261023_v3/runs/default/s42/oom_memory_snapshot.json)、[统计与分析](evidence/analysis.json) 已保留；快照采集于异常处理器，可能包含部分栈展开后的释放，不能把它当作每个 cudaMalloc 瞬间的完整现场。

单独使用 `max_split_size_mb=128` 将碎片降下来，却在第一次更新时触发了 **spconv/cumm 外部 TensorStorage 的 CUDA OOM**。该分配没有进入 PyTorch allocator 的 OOM/retry 路径（两个计数均为 0），Torch 缓存中又保留了大块空闲空间。独立进程直接处理上述最难增强也重现了 spconv 外部工作区 OOM，说明需要同时处理大块缓存回收。

最终进程内设置：

```text
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.7
torch.cuda.set_per_process_memory_fraction(1.0)
```

环境变量在导入 torch 前设置，memory fraction 在模型分配前设置。torch 2.0 的 native GC 路径受 `set_fraction` 条件控制，因此需要第二行来启用配置的缓存回收；**1.0 保留整卡容量**。0.7 是空闲缓存的回收触发阈值，实测 live allocated 可以超过该阈值。具体实现依据 [PyTorch v2.0.0 allocator 源码](https://github.com/pytorch/pytorch/blob/v2.0.0/c10/cuda/CUDACachingAllocator.cpp)。没有采用该版本不支持的 `expandable_segments`。

所有设置只作用于独立测试进程。Block checkpoint 继续减少激活存储，本轮 allocator 控制回收空闲缓存并限制碎片；没有修改 Tensor 数值、精度、网络结构或训练计算。

**逐项失败/成功对照**

下表各行使用同一六裁剪压力计划，seed 42、相同初始模型、原 optimizer/scheduler/loss、相同样本顺序与增强。默认与最终组合前六次成功更新的 `sample_digests` 全部一致。

| allocator 设置，seed 42 | 完成更新 | 失败阶段 | peak allocated / reserved GiB |
|---|---:|---|---:|
| 默认 native | 6/12 | 第 7 次 backward，Torch OOM | 19.253 / 22.898* |
| 仅 split128 | 0/12 | 第 1 次 backward，spconv/cumm OOM | 17.832 / 22.980* |
| 仅 gc70，fraction=1.0 | 6/12 | 第 7 次 backward，Torch OOM | 20.387 / 22.266* |
| **split128 + gc70，fraction=1.0** | **12/12** | 完成 | **20.518 / 21.236** |

`*` 是失败前捕获的 Torch allocator 峰值，不能视为该方案完成训练所需显存。所有表均为 GiB=2^30 bytes，**不包含全部 spconv/CUDA 外部分配**；实际完成与原始异常栈共同作为判断依据。完整每步 allocated、reserved、peak、耗时及 allocator 统计见 [运行汇总](evidence/runs_summary.json) 和同目录的 `*_steps.csv`。

**三个种子与完整输入保留**

从已验证的 25,000 个真实裁剪中，按点数选择 P10/P50/P90，并选择三个不同森林的最大裁剪：

| 角色 | 文件 | 原始点数 |
|---|---|---:|
| P10 | `L1N_951.npz` | 652,490 |
| P50 | `A1N_549.npz` | 1,227,066 |
| P90 | `G3W_1149.npz` | 1,653,047 |
| 全池最大 | `G2W_529.npz` | 2,677,003 |
| 另一森林最大 | `O1W_455.npz` | 2,280,704 |
| 第三森林最大 | `G4N_501.npz` | 2,102,583 |

每种模型种子执行同一尺寸切换顺序，24 个 micro-batch 各用独立增强种子，其中最大裁剪出现 8 次。包括大→小、小→大、最大→最大，以及 AdamW 状态建立后的最大尺寸回访。实际所有 72 个训练输入均保留全部点、实例和空间边界；没有合成点数、截取点云或改变既有采样/voxelization。

| 最终组合 | 完整更新 | 最大体素数 | peak allocated / reserved GiB | 第 2–12 步平均秒/更新 |
|---|---:|---:|---:|---:|
| seed 42 | 12/12 | 2,327,984 | 20.518 / 21.236 | 3.073 |
| seed 43 | 12/12 | 2,260,504 | 19.449 / 20.102 | 2.996 |
| seed 44 | 12/12 | 2,292,038 | 20.126 / 20.953 | 3.135 |

三个进程均满足：15 个 BN 层各恰好更新 24 次、running statistics 有限；AMP 跳步数为 0，scale=1024；AdamW 所有已建立状态的 step 均为 12，最终参数有限。每次更新后的存活 Point 数为 0，进程结束统计创建/释放数一致。原训练入口持有当前 loss 等局部对象，因此每步观测的 allocated 随最后一个输入大小变化；函数返回后的 allocated 均约 0.763–0.764 GiB，当前证据未显示跨步 Point 泄漏。

对最难增强另做新进程、原始 NPZ 输入、相同增强、固定初始模型的 cold-start 对照：

| 2,677,003 点 / 2,327,984 体素，累积=2 | 完成更新 | peak allocated / reserved GiB |
|---|---:|---:|
| 默认 allocator | 0/3，spconv/cumm OOM | 20.524 / 22.939* |
| 最终组合 | **3/3** | **21.204 / 21.732** |

最终组合第 2/3 次有效更新约为 5.47/5.59s。该数据与原固定增强不同，不能直接用它估计 allocator 的单项性能代价。可直接配对的 seed 42 第 2–6 步，同输入顺序下默认平均 2.322s，组合平均 2.890s，约增加 **24.5%**；该短序列包含尺寸切换和重新分配代价，不是长期稳态吞吐测量。

**固定输入的数值 A/B**

使用同一份冻结的完整输入/voxel-ID/v2p 映射、初始权重与 RNG，分别验证三个 allocator 设置。每个方案独立进程，保持原 FP16 autocast / Flash FP16，比较所有点的两个预测输出、459 个 unscale 后/clip 前梯度、45 个持久 buffer 和 CPU/NumPy/CUDA RNG。沿用上一轮接受阈值，没有为了本轮结果更改阈值。

| 对照 | 全局梯度相对 L2 | 最大绝对梯度差 |
|---|---:|---:|
| 106 万点，split128 对原 block_nested | 0.025835% | 0.00073242 |
| 106 万点，gc70 对原 block_nested | 0.025708% | 0.00073242 |
| 106 万点，最终组合对原 block_nested | 0.025714% | 0.00085449 |
| 最难增强，最终组合对 CPU byte-preserving 参考 | 0.041157% | 0.00097656 |
| 最难增强，CPU 参考自身重复 | 0.041158% | 0.00057983 |

以上 **5 组均通过**。预测最大绝对差和 loss 差为 0，全部 buffer 和 RNG 完全一致。最大尺寸原 GPU 参考无法完成，因此沿用已交叉验证的 byte-preserving CPU offload 作为数值参考；没有填造一条原 GPU 成功运行。梯度全局差与同尺寸参考自身重复相当。接近零的局部梯度可能有较大相对差，逐参数绝对/平方误差全部保存在 [数值汇总](evidence/comparisons_summary.json) 所指的完整 `comparison_*.json` 中，不能把全局阈值理解成每个参数均在 0.1% 内。

训练前六步的 loss 轨迹最大差约 0.00149，固定输入的前向 A/B 则完全一致。这与 GPU 反向非确定性造成的微小更新差异相符；没有验证长期训练轨迹逐位一致或最终精度收敛。

**实现、保护与复现**

新入口为 `tools/experiments/ptv3_memory_stability_v1.py`，管理模块为 `tree_learn/experiments/ptv3_memory_v1/stability.py`，策略为 `configs/experiments/ptv3_memory_v1/stability_policy.yaml`。它调用**未修改的 baseline `train.run`**，仅在本独立进程内替换模型 factory、挂接标量观测，并在 finally 恢复绑定。原按 semantic/offset 各自有效点数加权的累积损失、AdamW、scheduler、AMP unscale/clip/step 均沿用原逻辑。12 次更新位于第一 epoch，实际 lr=6.98e-5；没有更改原长期 scheduler 或用平均两个不同大小 crop 的 loss 替代原加权规则。

正式 sample plan 未改。私有压力计划只替换副本的前 24 项，保留原计划剩余部分以满足既有入口校验；本 CLI 固定最多执行 12 次更新，只有 prepare/run 两种操作。它输出的 `last.pth` 和 `smoke_final.pth` 为独立短测快照，`is_formal_final=false`，没有选择正式模型或使用评估集。

TreeLearn 核心 3 个文件、PTv3 baseline 19 个文件、原配置 16 个文件逐字节不变；三个正式 sample plan 及六个真实裁剪的 SHA256 均再次校验，见 [integrity.json](evidence/integrity.json)。main 仍为 `fd240ce7caa4c444fe3418aca454dc578bc557d4`，baseline 仍为 `855f831c01089261b93658fa2d9a4507bc43be27`。

首个准备目录因 JSON 科学计数法经 YAML 1.1 读取后发生类型变化而被一致性检查拦截，未进入训练。改为原生 YAML 并验证 round-trip 类型后才执行上述测试；原失败日志保留在 evidence 的 `stability_20261023_v1` 中。最终 [12 项回归测试](evidence/unit_tests.log) 全部通过，包括该类型问题、计划边界、外部 OOM 分类和原 checkpoint 回归。

完整原始数据、Tensor dump 和短测 checkpoint 在：

```text
/home/szu-stu/projects/zrx/code/TreeLearn/work_dirs/ptv3_memory_v1_20261023/stability_20261023_v3
```

Git 中附完整日志、失败栈、allocator snapshot、计划的可读样本序列、manifest、逐步 JSON/CSV、逐参数对照和 SHA inventory；大 NPZ/PTH 保留在上述目录，SHA 由计划、fixture manifest、数值比较和 checkpoint provenance 记录。证据采集脚本为本目录 `collect_evidence.py`，运行记录汇总见 [runs_summary.json](evidence/runs_summary.json)，路径与校验信息见 [artifact_inventory.json](evidence/artifact_inventory.json)。

复现时使用全新目录；下面每个 run 固定 12 次更新，推荐设置需显式选择：

```bash
cd /home/szu-stu/.codex/worktrees/ptv3-treelearn-baseline/TreeLearn
env -u PYTORCH_CUDA_ALLOC_CONF OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/szu-stu/miniconda3/envs/TreeLearn/bin/python \
  tools/experiments/ptv3_memory_stability_v1.py prepare \
  --plan-dir work_dirs/ptv3_memory_v1/stability_repro

env -u PYTORCH_CUDA_ALLOC_CONF OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/szu-stu/miniconda3/envs/TreeLearn/bin/python \
  tools/experiments/ptv3_memory_stability_v1.py run \
  --plan-dir work_dirs/ptv3_memory_v1/stability_repro \
  --seed 42 --allocator split128_gc70
```

对 seed 43、44 分别执行后一条命令。不要把这个私有压力计划用于正式训练；它是针对尺寸切换的测试集合。后续仍需验证更长运行、更多增强极值、跨 epoch、断点恢复和精度收敛，本轮结果只覆盖所列短稳定性条件。
