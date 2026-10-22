# PTv3 / TreeLearn 实施、资源与公平性报告

本次交付范围为 P0–P2，以及 P3/P4 的可执行入口。没有启动正式长期训练，没有产生 E0/E1 的正式单木分割成绩，也没有据此作 Backbone 优劣结论。

## 代码和文件

- TreeLearn 基线：`fd240ce7caa4c444fe3418aca454dc578bc557d4`，通过 fetch 和远端引用核验的 `origin/main`。
- 独立分支：`codex/ptv3-treelearn-baseline`；worktree：`/home/szu-stu/.codex/worktrees/ptv3-treelearn-baseline/TreeLearn`。
- 实现提交：`9521588`；模型/训练接口冻结提交：`f80aab665514596caeda07883a3c39bb2610ffef`。100-step、恢复和五组推理证据的公共 Python 源码 SHA256 为 `cdda8c80992985dfeb8ee56f3aed917977d1f647780cafa1874527ab2e97c612`。后续补充的聚合门禁仅修改 aggregate.py 及对应测试，未改变模型、数据、训练或推理运算；其实际源码哈希见 [聚合验证](evidence/aggregation_validation.json)。每份运行证据保留当时的完整哈希，不追改为交付版本。
- PTv3 上游：`Pointcept/PointTransformerV3@3229e9b7de1770c8ad17c316f8e349982de509f8`，MIT；model、serialization、LICENSE 的上游与 vendor SHA256 逐文件一致，见 [上游记录](evidence/ptv3_upstream.json)。

原有模型、数据集、工具与默认配置没有改动；原工作目录仍处于历史实验分支。修改文件完整清单见 [changed_files.txt](changed_files.txt)。主要新增内容：

| 位置 | 内容 |
| --- | --- |
| `tree_learn/experiments/ptv3_treelearn/` | 模型接口、原版任务复用、数据和固定计划、训练/恢复管理、推理边界处理、评价、诊断、聚合、上游源码 |
| `tools/experiments/ptv3_treelearn.py` | audit-data / generate-crops / plan / train / infer / evaluate / diagnose / aggregate |
| `configs/experiments/ptv3_treelearn/` | E0-main-hais、E0-scratch、E1、两种 E2，以及完整输入、microbatch、checkpointing、资源变体配置 |
| `tests/ptv3_treelearn/` | 21 项契约测试与实际 CUDA 断点恢复测试 |
| `docs/experiments/ptv3_treelearn/` | 源码审计、本报告、可提交的小型证据文件 |

配置、四组训练/评价/诊断命令和三 seed 聚合入口集中在 [README](../../../configs/experiments/ptv3_treelearn/README.md)。原始行为和共同适配的逐项区分见 [源码审计](source_audit_zh.md)。

## 已实际执行的测试

1. **22 项契约测试全部通过**，最新测试主体 2.792s。覆盖 E0 原类/参数一致性、4 常数特征、重复体素/多 batch/交错 batch 映射、PTv3 点序、GT 的 11/12 点边界、ignore/masks、Loss/空 mask、梯度、HAIS 加载门禁、原版正常 grouping/ensemble、空 seed/全 noise/少参考点、评价极端输入、未知标签 LAZ 往返、checkpointing 梯度和 BN、streaming tile 数组/元数据、无损 NPZ/SHA、恢复日志回滚和拒绝 smoke 正式聚合。新增的聚合测试使用明确标注的合成记账 fixture，验证三 seed 正常统计，并拒绝跨轮预测路径、改写预测文件、不同 grouping/config/data；它不冒充正式训练。另已对两份实际 E2 测试片结果运行聚合门禁。见 [测试日志](evidence/final_unit_tests.log)。
2. 实际 CUDA 共享前缀测试：连续训练至 update 4，与同一 update 2 checkpoint 重启后训练至 update 4 比较。恢复前所有保存状态逐值一致，样本序列、LR、Scaler、RNG 和游标一致；后续模型最大绝对差值 **0.0001275615**，低于预先规定的 0.0002 容差，见 [恢复记录](evidence/final_resume.json)。**继续计算的模型和优化器不是逐位相同**；不能用这次短测试保证长程数值轨迹逐位重放。
3. 冻结模型/训练版本分别完成 E0-scratch、E0-main-hais、E1 各 **100 updates / 100 次实际 optimizer.step**，全部梯度有限、没有 AMP 跳步。三组全部 100 步的增强输入、GT、masks 摘要一致，公共公平性指纹一致。真实 G1N crop 原始 1,060,733 点，每 microbatch 共同选取 400,000 点；200 个增强样本的实例保留率均为 1.0，XY 极值范围逐项相等。见 [100 步记录](evidence/final_smoke100.json)。该旧 crop 被显式标为 unverified_smoke，只用于工程测试，不能用于正式训练。

短程 Loss 下降只说明优化过程可执行；不同初始条件、仅一个旧 crop 的结果不能用于科学性能比较。

## RTX 4090 实测

环境为 RTX 4090 24GB（torch 报告可用总容量 23.53GiB）、Python 3.10.19、torch 2.0.0/CUDA 11.8。FlashAttention 2.3.6 安装在 worktree 的独立 runtime 目录，没有升级 Conda 的 torch、spconv 或 CUDA。具体包和驱动见 [环境记录](evidence/environment.json)，wheel 来源和 SHA256 见 README。

### 最终公共资源配置，100 次更新

每次有效 batch=2，microbatch=1、累积=2；AMP 初始 scale=1024；共同 400k 点上限；PTv3 采用标准宽度/深度并重算 Attention/MLP。下表计时剔除第一步，包含同步的 forward/backward/梯度检查/optimizer 部分；完整 update 另含加载和摘要。

| 模型 | 参数数 | 核心单步均值±标准差，s | 含数据的单步均值，s | Peak allocated / reserved，GiB |
| --- | ---: | ---: | ---: | ---: |
| E0-scratch | 30,106,981 | 0.14748 ± 0.01823 | 0.32810 | 3.1085 / 4.4492 |
| E0-main-hais | 30,106,981 | 0.13948 ± 0.00728 | 0.31331 | 3.1087 / 4.4492 |
| E1 PTv3-resource-limited | 46,167,173 | 0.68204 ± 0.03263 | 0.70286 | 10.5726 / 12.2305 |

这里用实测同步单步成本作计算成本估计，没有编造稀疏网络 FLOPs。PTv3 的核心单步时间约为 E0-scratch 的 4.62 倍。缓存复用、森林密度、数据盘读取和并行数据准备会影响时延；不能将本表直接当作全数据吞吐或正式训练时长保证。

系统盘不能容纳全部正式 checkpoint：按当前参数/AdamW 状态和每 20 epochs 留档估算，九个正式 run 的周期 checkpoint 约为 200GB 量级，尚不含整林输出和重复副本。已将尚未使用的 `work_dirs/ptv3_treelearn/{formal,inference,evaluation}` 链接到项目 HDD 的独立 namespace；没有在这些目录启动训练。新裁剪池也位于该数据盘，具体路径见 README。

### 完整真实裁剪的可行性边界

| 实测输入/设置 | 结果 |
| --- | --- |
| A1N 511,276 点，完整输入，PTv3 物理 batch=2 | 独立进程仍 OOM；peak allocated 22.84GiB，详见 [记录](evidence/gpu_full_e1_isolated.json) |
| 相同 A1N，microbatch=1、累积=2 | 完成 2 次有效更新；peak allocated 12.76GiB |
| G1N 1,060,733 点，完整输入，microbatch=1、累积=2、修复后 tensor checkpointing | 完成 10 次更新；peak allocated 17.61GiB |
| G2W 2,717,262 点，完整输入，microbatch=1、累积=2、修复后 tensor checkpointing | 首次前向仍 OOM；peak allocated 21.94GiB，详见 [记录](evidence/gpu_largest_micro_checkpoint.json) |
| G1N 1,060,733 点，原 main U-Net 物理 batch=2、完整输入 | 完成真实前向、反向和一次更新；peak allocated 8.32GiB，冷启动首步 32.16s，不能当稳态吞吐，详见 [记录](evidence/e0_full_input.json) |
| G2W 2,717,262 点，PTv3 eval/no_grad、无点数上限、main FP32 推理 | 完成 `[N,2]` / `[N,3]` 有限输出；1.80s，allocated/reserved 8.51/15.23GiB，详见 [记录](evidence/largest_inference.json) |

因此，本机不能承诺标准完整输入配置覆盖全部 crop。`*_resource.yaml` 是明确命名的独立方案，不能被描述为原版 main 全点集复现。未缩小 PTv3 网络层数、通道或 patch；改变的是所有组共同使用的训练输入上限与执行方式。

main 的固定 50m XY grid 另有实际覆盖问题：200 个 G1N 增强输入中有 27 个越界。资源接口对三个训练组共同启用规范化体素 ID 与覆盖实际输入的 grid 策略；U-Net 仅扩展越界轴，普通输入仍保留原固定尺寸，正常前向已验证一致。该适配和原始默认入口分开记录。

## 数据、初始化和隔离

18 个 raw LAZ 的 SHA256、发布平台 MD5、点数、实例点数分布与 Z Extent 已核查，合计 6,665 个正标签实例；来源和所有逐文件哈希在 [source_manifest.json](evidence/source_manifest.json) 与 [published_checksums.json](evidence/published_checksums.json)。保留 raw 标签，没有按 10m 高度阈值重标注。

已有 25,000 个 crop 缺少可核验的完整生成 seed，因此在独立数据盘目录重新生成。实际位置为 `/home/szu-stu/projects/zrx/code/TreeLearn/data/ptv3_treelearn_baseline_20261023/v1`，worktree 通过 `data/experiments/ptv3_treelearn/v1` 访问；不覆盖历史裁剪。生成保持 main 的 0.1m 下采样、0.6m Verticality、occupancy、候选选择和 35m 窗口定义。并行保存只改变执行方式；完整顺序随机选择计划先冻结，每 crop 保留写时 SHA256、中心、角度、点数、未增强监督点数、实例点数和 Z Extent。训练时另记实际增强后的点数和 masks。

| 初始化 | SHA256 与核验 |
| --- | --- |
| HAIS | `67cacb699d1a6fffb53ccb60e0d35d072495fa479771b6a46e16e2348f77702e`；从 main training guide 的 Google Drive 来源独立重下，与本地文件逐字节哈希一致 |
| E2-default 20241213 | `5df2f92828f92755bc12e114eaebe83f7ecea94a74c25a6170b68844cc5e19bb`；独立下载，发布 MD5 一致，严格完整加载 |
| E2-with-small 20241213 | `513c98d44b5252bb215599b52cdc7a521ac376621738529ac79d0936b24ce1c6`；发布 MD5 一致，独立命名、严格完整加载 |

HAIS 的必需 U-Net/output 张量覆盖率为 100%，但**不是全模型初始化**：原输入卷积 6→4 通道、语义末层 20→2 类形状不符等情况被单列。每一个成功、缺失、形状不符、多余/跳过键及形状见 [完整加载报告](evidence/hais_initialization.json)。E0-scratch/E1 不加载预训练。

L1W raw 为 85,289,646 点、200 个正标签，评价 GT 为 20,434,648 点、156 个完整标注实例。Wytham 本地输入和 GT 均为 54,592,456 点、877 个正标签；已核对字段、坐标范围、SHA256 和公开 GT 的 MD5。未知/边界点按 main 的 LAS/LAZ 标签约定处理，不引入论文中的额外边界裁剪。Wytham 为跨域开发评价，不用于训练、调参、早停或 checkpoint 选择。

## 真实森林推理闭环

E0/E1/E2 的小型真实 A1N 森林测试片来源为 27,822 个点、2 个可见实例，属于工程 smoke，无法代表完整森林精度。最终代码采用共同 `--compact-pointwise --stream-tiles` 接口，仍调用 main 的 ensemble、Semantic/Verticality/Offset-Z seed、XY HDBSCAN、3D shifted 5NN、整林标签恢复和 LAZ 输出。保存的 backbone features 可省略，预测和 Verticality 完整保留；streaming tile 的逐数组、dtype、中心、JSON 与 main 一致。

五个独立组别（E0 两组、E1、E2-default、E2-with-small）在最终源码上均完成该 23-tile 测试片的推理、LAZ、评价和诊断；逐组配置/权重/GT/源码哈希、实际时间、显存和工程指标见 [闭环记录](evidence/final_pipeline.json)。三组训练模型使用上面的 100-update smoke_final，不能冒充完整预算 final。最大旧 crop 的完整 PTv3 点级推理通过只说明该样本可行，不保证所有新 crop 或整片森林均有相同峰值。

另外，已经实际完成 **E2-default 的完整 L1W**：fresh voxel/feature/tiling → 1,221 tiles → 原版实例生成 → 恢复 85,289,646 点 LAZ → 对齐评价 GT。流程时间 3,027.78s（不含之后单独评价），GPU peak allocated/reserved 为 2.83/8.72GiB；原版非流式 tiler 的采样 host VmHWM 约 70.74GiB。今后的整林运行建议共同使用 streaming 和 compact 选项，仍须记录实际 host/GPU 峰值。

| E2-default L1W 指标 | 实测值 |
| --- | ---: |
| TP / FP / FN | 156 / 2 / 0 |
| Detection Precision / Recall / F1 | 0.987342 / 1.000000 / 0.993631 |
| Completeness / Omission / Commission | 1.000000 / 0.000000 / 0.012658 |
| Coverage | 0.980957 |
| Semantic Tree Recall | 0.999719 |
| Offset XY / XYZ Error | 0.330714m / 0.417512m |
| 初始 / 最终实例数 | 305 / 305 |
| Seed 数 | 325,544 |
| Z Extent <10m / 10–20m / ≥20m GT 数 | 0 / 19 / 137，各非空分箱 recall=1.0 |

main Commission 规则只计算满足对 GT 重叠条件的未匹配预测，因此最终 305 个实例不等于 TP+FP=158。Offset 诊断使用独立 benchmark GT 的**整片可见点**按 main getOffset 构造 anchor，与训练的增强 crop anchor 区分；最近邻对齐距离中位数 0.01677m、95% 为 0.05193m。

该完整 L1W 流程在首个实现提交之前运行，原 manifest 记录了 checkpoint、数据、环境和 dirty Git 状态，**没有捕获运行时公共代码 SHA256**。保留原始记录并明确此限制，不能追补一个臆造的执行代码哈希，也不把它作为冻结正式成绩。指标仅是已执行闭环的 E2 工程证据，见 [指标](evidence/e2_default_L1W_metrics.json)、[诊断](evidence/e2_default_L1W_diagnostics.json)、[流程记录](evidence/e2_default_L1W_pipeline.json)。正式发表应在最终交付 commit 上重跑。

## 公平性审计和正式实验前工作

| 项目 | 协议 |
| --- | --- |
| 主要架构比较 | E0-scratch vs E1；E0-main-hais 的预训练差异单独解释 |
| 训练数据/增强/监督 | 同 18 forests、同 25,000 crop、同 seed 计划，main 完整增强→getOffset/masks→共同资源抽样 |
| 输入/Head/Loss | 同 4 常数通道、两层 MLP 定义、50×CE+欧氏 Offset；输出 C=32/64 的宽度和参数差异公开 |
| 共同训练管理 | 同 AdamW/Scheduler、先 unscale 再 clip(1)、AMP 初始 scale=1024、固定 250,000 次更新尝试；跳步单列 |
| microbatch 差异 | 有效点加权归一化；BN 仍不同于物理 batch=2，不能声称逐操作等价 |
| 资源差异 | 完整输入与共同 400k 点变体分开；PTv3 checkpointing 不改网络规模；推理不限制点数 |
| 后处理/评价 | 固定 main HDBSCAN、阈值、3D shifted 5NN、IoU>0.5 和 Commission，不为 PTv3 单独调参 |
| 模型选择 | 仅完整预算终点 final.pth；L1W/Wytham 不进入训练或选择 |
| 多 seed 统计 | 支持 42/43/44；正式聚合要求全部三个 seed、final SHA、GT/data/plan/公共协议匹配，标准差 ddof=1；评价预测必须来自索引中的推理目录且当前 SHA 匹配，分组/裁块/评价规则一致，同组跨 seed 不混配置、同组同 seed 跨域不换 checkpoint |
| E2 解释 | default/with-small 分开；公开来源不足以排除历史训练与 L1W/Wytham 重叠，只作成熟模型参考 |

正式长期训练仍需用户另行授权。授权前还应完成全数据 500-update Pilot，检查多密度 crop 的梯度、显存、AMP 跳步和 checkpoint 恢复；确认接受共同资源变体与 CUDA 非逐位重放限制。正式训练后才能在同一最终 commit、相同推理选项和固定 final 上重跑 L1W/Wytham、诊断 Semantic→Offset→Seed→HDBSCAN→5NN，并聚合三 seed。完整 Wytham 和 E2-with-small 的整林成绩本次未运行，不能填入预期数字。
