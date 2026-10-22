# PTv3 / TreeLearn main 对照实验入口

所有命令在本独立 worktree 根目录运行。原版 `tools/training/train.py`、`tools/pipeline/pipeline.py` 和 `configs/*` 默认配置未修改；实验入口为 `tools/experiments/ptv3_treelearn.py`。

源码与公平性审计见 [source_audit_zh.md](../../../docs/experiments/ptv3_treelearn/source_audit_zh.md)，实际运行记录见同目录下的实施报告。完整预算训练需要用户另行授权，本次仅执行短程测试与数据准备。

## 环境和路径

本机 Python：`/home/szu-stu/miniconda3/envs/TreeLearn/bin/python`（3.10.19 / torch 2.0.0 / CUDA 11.8）。以下 shell 变量只用于方便执行命令：

```bash
TREELEARN_PYTHON=/home/szu-stu/miniconda3/envs/TreeLearn/bin/python
TREELEARN_RUNNER=tools/experiments/ptv3_treelearn.py
```

本 worktree 的 `data/source` 链接到已有 `TreeLearn/data`，仅作为数据来源。部署到其他主机时，在新的 worktree 中建立相同布局：

```bash
mkdir -p data
ln -s /absolute/path/to/original/TreeLearn/data data/source
```

所有本轮生成数据和权重存放在 `data/experiments/ptv3_treelearn`，日志、checkpoint、推理与评价存放在 `work_dirs/ptv3_treelearn`。CLI 拒绝覆盖已有 run 输出目录。

本机的大型新裁剪池通过 `data/experiments/ptv3_treelearn/v1` 链接到项目数据盘的独立目录 `/home/szu-stu/projects/zrx/code/TreeLearn/data/ptv3_treelearn_baseline_20261023/v1`。系统盘只有数十 GB 余量，完整生成前应选择有数百 GB 空间的数据盘；本轮未覆盖 `data/source/train/random_crops`。

FlashAttention 2.3.6 的 CPython 3.10 / torch 2.0 / CUDA 11.8 wheel 安装在 worktree 的独立目录，未修改 Conda 环境。CLI 自动加载该目录。重建同一运行环境：

```bash
mkdir -p work_dirs/ptv3_treelearn/runtime/wheels
curl -fL --retry 3 -o work_dirs/ptv3_treelearn/runtime/wheels/flash_attn-2.3.6+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl \
  'https://github.com/Dao-AILab/flash-attention/releases/download/v2.3.6/flash_attn-2.3.6+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl'
echo '1c918d84ff042582d547e4c9c7ec24ec9f3975589ca28cc4e6cbd15486b74f2a  work_dirs/ptv3_treelearn/runtime/wheels/flash_attn-2.3.6+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl' | sha256sum -c -
"$TREELEARN_PYTHON" -m pip install --no-deps --target work_dirs/ptv3_treelearn/runtime/packages \
  work_dirs/ptv3_treelearn/runtime/wheels/flash_attn-2.3.6+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

不使用 pip 安装/升级 torch、CUDA 或 spconv。`TREELEARN_RUNTIME_PACKAGES` 可显式选择另一个隔离的运行目录，实际包版本进入 environment.json 和恢复身份。

## 配置与组别

| 配置 | 用途 |
| --- | --- |
| `e0_main_hais.yaml` | main U-Net，按报告部分加载 HAIS，原物理 batch=2、完整点集。 |
| `e0_scratch.yaml` | main U-Net，全随机初始化，原物理 batch=2、完整点集。 |
| `e1_ptv3.yaml` | 标准 PTv3 宽度/深度、4 常数输入、原版两层 Head 与 Loss，原物理 batch=2。 |
| `e2_official.yaml` | 官方 `model_weights_20241213.pth`，只评价。 |
| `e2_official_with_small.yaml` | 独立命名的官方 `with_small_20241213`，只评价。 |
| `*_micro.yaml` | 完整点集，microbatch=1、累积=2；保留 BN 差异说明。 |
| `e1_ptv3_checkpoint.yaml` / `e1_ptv3_micro_checkpoint.yaml` | 额外使用只重算 Attention/MLP 的 tensor checkpointing。 |
| `*_resource.yaml` | 共同最大 400k 点，microbatch=1、累积=2、确定性体素编号、覆盖增强后输入的 grid；PTv3 保持标准网络规模并使用 tensor checkpointing。所有组先在完整 crop 生成 GT/masks，再共同抽样。 |
| `e1_ptv3_determinism_audit.yaml` | 独立的数值确定性诊断变体；要求额外隔离的 FlashAttention≥2.4，不混入默认性能统计。 |

主要比较为 E0-scratch vs E1。E0-main-hais 仅用于当前 main 配方参考；E2 的历史训练条件和数据重叠未被完全排除，不能用于公平 Backbone 归因。

## P0 数据生成和固定计划

```bash
"$TREELEARN_PYTHON" "$TREELEARN_RUNNER" audit-data
"$TREELEARN_PYTHON" "$TREELEARN_RUNNER" generate-crops --seed 42
# 本轮拥有的生成目录如中断，可以验证缓存后恢复；不同 seed/config 会报错。
"$TREELEARN_PYTHON" "$TREELEARN_RUNNER" generate-crops --seed 42 --resume-generation
# voxel/features/occupancy 缓存已完整后，可在同一独立目录恢复并行保存。
# 先复现顺序 main 的完整随机选择计划，再并行写文件；不改变候选和样本选择。
"$TREELEARN_PYTHON" "$TREELEARN_RUNNER" generate-crops --seed 42 --resume-generation --generation-workers 4
for experiment_seed in 42 43 44; do
  "$TREELEARN_PYTHON" "$TREELEARN_RUNNER" plan \
    --crop-manifest data/experiments/ptv3_treelearn/v1/crop_manifest.json \
    --seed "$experiment_seed" --output "data/experiments/ptv3_treelearn/plans/s${experiment_seed}.npz"
done
```

第一次生成必须使用不存在的独立目标目录。仅在中断后执行恢复命令，不能同时启动两个生成任务。完整生成沿用 main 的 voxelize / features / occupancy / candidates / 裁剪提取。固定森林遍历顺序；恢复 occupancy cache 时重放原本应消耗的 RNG 抽样，复用已完成森林时核查全部 crop 中心和角度与重放计划一致。文件采用无损 NPZ 压缩、写时 SHA256 和数组审计；已完成的未压缩文件不重写。每片森林记录逐 crop SHA256、元数据、中心、范围、未增强内区的语义/Offset 有效点数、实例点数和 Z Extent 分布；训练日志另记实际增强后的监督点数。只有 verified 且恰好 25,000 个裁剪的 manifest 才能用于计划。

## P2 测试和真实 GPU Smoke

```bash
PYTHONPATH=work_dirs/ptv3_treelearn/runtime/packages:. \
  "$TREELEARN_PYTHON" -m unittest discover -s tests/ptv3_treelearn -v
"$TREELEARN_PYTHON" tests/ptv3_treelearn/run_resume_check.py
```

恢复测试同时报告“保存状态是否逐值恢复”和“CUDA 继续运算是否逐位一致”，并保留后续运算的最大差值。当前预设 CUDA 数值容差为 2e-4；位级一致性单列，不得仅靠 RNG 一致宣称位级重放。测试每次创建独立时间戳目录，`TREELEARN_RESUME_OUTPUT` 可以指定不存在的输出目录。

允许将历史单块 crop 显式用于 ≤100 updates 的工程 smoke，禁止将其用于 pilot/formal；run manifest 会标记 unverified_smoke：

```bash
"$TREELEARN_PYTHON" "$TREELEARN_RUNNER" train \
  --config configs/experiments/ptv3_treelearn/e1_ptv3_resource.yaml \
  --phase smoke --max-updates 100 --seed 42 \
  --smoke-crop data/source/train/random_crops/npz/G1N_0.npz \
  --work-dir work_dirs/ptv3_treelearn/my_e1_smoke
```

## P3：已核验完整数据上的短程入口

同一配置分别用 E0-main-hais、E0-scratch、E1 执行，依赖上面的正式裁剪 manifest 与冻结计划：

```bash
for experiment_group in e0_main_hais e0_scratch e1_ptv3; do
  "$TREELEARN_PYTHON" "$TREELEARN_RUNNER" train \
    --config "configs/experiments/ptv3_treelearn/${experiment_group}_resource.yaml" \
    --phase smoke --seed 42 --work-dir "work_dirs/ptv3_treelearn/${experiment_group}_smoke100"
  "$TREELEARN_PYTHON" "$TREELEARN_RUNNER" train \
    --config "configs/experiments/ptv3_treelearn/${experiment_group}_resource.yaml" \
    --phase pilot --seed 42 --work-dir "work_dirs/ptv3_treelearn/${experiment_group}_pilot500"
done
```

不设置 `--max-updates` 时 smoke=100、pilot=500；结果不能作为论文成绩。原始完整点集对照只需选择不带 resource 后缀的配置，实际可行性以资源报告为准。

## P4：另行授权后使用的长期训练命令

以下命令仅提供入口，本次不执行。每个组/seed 必须选择同一个输入资源协议，保持 1,000 epochs×500 examples / effective batch 2 = 250,000 updates。不得将 resource 和完整输入的结果合并统计。

```bash
for experiment_seed in 42 43 44; do
  for experiment_group in e0_main_hais e0_scratch e1_ptv3; do
    "$TREELEARN_PYTHON" "$TREELEARN_RUNNER" train \
      --config "configs/experiments/ptv3_treelearn/${experiment_group}_resource.yaml" \
      --phase formal --seed "$experiment_seed" \
      --work-dir "work_dirs/ptv3_treelearn/formal/${experiment_group}_s${experiment_seed}"
  done
done
```

单 GPU 应串行运行；不要在本机同时启动多个训练任务。恢复正式训练须使用同一配置、phase、seed 和工作目录：

```bash
"$TREELEARN_PYTHON" "$TREELEARN_RUNNER" train \
  --config configs/experiments/ptv3_treelearn/e1_ptv3_resource.yaml \
  --phase formal --seed 42 --work-dir work_dirs/ptv3_treelearn/formal/e1_ptv3_s42 \
  --resume work_dirs/ptv3_treelearn/formal/e1_ptv3_s42/last.pth
```

## 四个实验组的统一推理、评价和诊断

E0-main-hais / E0-scratch / E1 的 `--checkpoint` 使用对应完整预算 `final.pth`；E2 使用独立命名的已发布权重。示例：

```bash
# E0-main-hais
"$TREELEARN_PYTHON" "$TREELEARN_RUNNER" infer --config configs/experiments/ptv3_treelearn/e0_main_hais_resource.yaml \
  --checkpoint work_dirs/ptv3_treelearn/formal/e0_main_hais_s42/final.pth \
  --forest data/source/pipeline/L1W/forest/L1W.laz --output work_dirs/ptv3_treelearn/inference/e0_hais_L1W_s42
# E0-scratch
"$TREELEARN_PYTHON" "$TREELEARN_RUNNER" infer --config configs/experiments/ptv3_treelearn/e0_scratch_resource.yaml \
  --checkpoint work_dirs/ptv3_treelearn/formal/e0_scratch_s42/final.pth \
  --forest data/source/pipeline/L1W/forest/L1W.laz --output work_dirs/ptv3_treelearn/inference/e0_scratch_L1W_s42
# E1
"$TREELEARN_PYTHON" "$TREELEARN_RUNNER" infer --config configs/experiments/ptv3_treelearn/e1_ptv3_resource.yaml \
  --checkpoint work_dirs/ptv3_treelearn/formal/e1_ptv3_s42/final.pth \
  --forest data/source/pipeline/L1W/forest/L1W.laz --output work_dirs/ptv3_treelearn/inference/e1_L1W_s42
# E2-default（不训练）
"$TREELEARN_PYTHON" "$TREELEARN_RUNNER" infer --config configs/experiments/ptv3_treelearn/e2_official.yaml \
  --checkpoint data/experiments/ptv3_treelearn/weights/model_weights_20241213.pth \
  --forest data/source/pipeline/L1W/forest/L1W.laz --output work_dirs/ptv3_treelearn/inference/e2_default_L1W
# E2-with-small（独立参考，不与 default 合并）
"$TREELEARN_PYTHON" "$TREELEARN_RUNNER" infer --config configs/experiments/ptv3_treelearn/e2_official_with_small.yaml \
  --checkpoint data/source/model_weights/model_weights_with_small_20241213.pth \
  --forest data/source/pipeline/L1W/forest/L1W.laz --output work_dirs/ptv3_treelearn/inference/e2_small_L1W

"$TREELEARN_PYTHON" "$TREELEARN_RUNNER" evaluate \
  --prediction work_dirs/ptv3_treelearn/inference/e1_L1W_s42/results/full_forest/L1W.laz \
  --gt data/source/benchmark/L1W_voxelized01_for_eval.laz \
  --output work_dirs/ptv3_treelearn/evaluation/e1_L1W_s42
"$TREELEARN_PYTHON" "$TREELEARN_RUNNER" diagnose \
  --pointwise work_dirs/ptv3_treelearn/inference/e1_L1W_s42/results/pointwise_results/pointwise_results.npz \
  --pipeline-manifest work_dirs/ptv3_treelearn/inference/e1_L1W_s42/pipeline_manifest.json \
  --gt data/source/benchmark/L1W_voxelized01_for_eval.laz \
  --output work_dirs/ptv3_treelearn/evaluation/e1_L1W_s42/diagnostics.json
```

Wytham 的同样命令只替换 forest=`data/source/pipeline/wytham/forest/wytham_vox0.1.laz`、GT=`data/source/benchmark/wytham_vox0.1.laz`，以及独立 output 名称。所有 grouping、5NN、匹配和 Commission 配置固定取 main。整片森林的 host RAM/推理显存仍须在正式评价前单独测量；小测试片闭环不能冒充完整 L1W/Wytham benchmark。

完整森林可统一增加 `--compact-pointwise --stream-tiles`：前者只省略保存的 backbone features，保留全部预测和 Verticality；后者只改变 tile 生成的内存执行方式。两项均已验证不改变 tile 点集或实例规则。三个正式训练 seed 的推理分别明确传入 `--seed 42`、`--seed 43`、`--seed 44`，并为所有模型采用相同执行选项。

正式指标聚合使用 `aggregate --index /path/to/index.json --output /path/to/report.json`。index 是记录列表，每项含 `group`（如 E0-scratch / E1-ptv3）、`domain`、`seed`、`metrics`（metrics.json 路径）、`pipeline_manifest`、`training_run`。聚合器要求完整 250,000-update formal final 的 SHA 与推理权重一致，要求三个 seed 齐备并核对数据/计划/公共协议，不接受 smoke/pilot；按 seed 列出指标，标准差使用 ddof=1。E2 独立分组，seed=null、无需 training_run，不能与 E0/E1 合并归因。
