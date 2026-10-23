# PTv3 完整裁剪正式训练 v1（2026-10-23）

用户最新指令为“直接开始长期训练”，已替代此前不启动长期训练的限制。本轮启动 E1-PTv3 seed 42；没有启动其他实验组、其他种子或独立 500-step Pilot。

正式预算为 1,000 epochs × 500 samples / effective batch 2 = **250,000 updates**。从随机初始化开始，不加载短测 checkpoint。沿用已验证的 25,000 个完整 35m 裁剪和原始 `plans/s42.npz`；不使用只含六裁剪的压力测试计划。原始计划 SHA256：`210d4b44bb6d6e0f190e525d9636867acf3c989183329815d790914712a5b688`；裁剪 manifest SHA256：`0c8a2046d6d17c706bff9bdcb81dfa075bf262b9acf2e9e83df1b6d1421dd66b`。

新配置 `configs/experiments/ptv3_memory_v1/e1_full_formal.yaml` 对原完整输入 microbatch 配置仅增加适配器、allocator 和 checkpoint 管理元数据。physical batch=1、accumulation=2；GT、有效点数加权 loss、模型通道/深度/patch/顺序、voxel_size、精度、AdamW、epoch scheduler 和 augmentation 均不变。微批累积的 BN 与 physical batch=2 存在既有差别，不能称为物理 batch=2 的逐操作复现。继续使用经过输出、梯度、BN/RNG 和三种子短测验证的 `block_nested`，长期稳定性与收敛仍由本次实测确定。

进程在 torch 导入前配置 `max_split_size_mb:128,garbage_collection_threshold:0.7`，模型分配前设置 memory fraction=1.0；不减少整卡容量。正式入口没有最大点数或缩小模型的开关。每步额外记录体素数、BN 次数和 allocated/reserved/累计 peak；标量观测不保留逐步 Tensor。AMP 溢出沿用原训练器的 GradScaler 处理并如实记录。

使用未修改的 `tree_learn.experiments.ptv3_treelearn.train.run`，通过进程局部 factory 接入优化。原先的**全计划数据 SHA256 预检保持执行**：约 473 GB、25,000 文件，完成后才建立模型并开始更新。`run_status.json` 区分 `verifying_data`、`building_model`、`training`、`complete` 与失败状态，不能把提交后台作业等同于完成训练。

每 50 次更新原子替换 `recovery.pth`；每 epoch 更新 `last.pth`；每 20 epochs 保留 `epoch_N.pth`；只有完整预算结束才写 `final.pth`。`latest_checkpoint.json` 记录实际最新路径、global update 与 SHA256。状态格式沿用原训练器，含 model/optimizer/scheduler/scaler/RNG/样本游标/代码与数据身份；恢复不允许身份变更。周期文件保持原命名和内容；滚动恢复避免生成 5,000 份大快照。OOM 保留完整异常、allocator stats 和 snapshot，不自动更改输入或重试不同模型。

独立运行根目录：

```text
/home/szu-stu/projects/zrx/code/TreeLearn/work_dirs/ptv3_full_formal_v1_20261023_s42
```

`source/` 为启动 commit 的独立本地 Git clone，数据与隔离 FlashAttention runtime 通过软链接只读使用，后续开发目录的修改不会改变运行源文件。`run/` 保存训练、checkpoint 与日志；`run_control/` 保存训练目录创建前的启动信息；根目录 `launch.json` 和 `console.log` 记录后台进程。该进程独立于本轮工具调用存活；没有配置静默自动重启或其他实验队列。

在冻结的 source 目录中启动/恢复（恢复时加 `--resume`，路径取 `latest_checkpoint.json`）：

```bash
env -u PYTORCH_CUDA_ALLOC_CONF OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/szu-stu/miniconda3/envs/TreeLearn/bin/python -u \
  tools/experiments/ptv3_memory_train_v1.py --seed 42 \
  --work-dir /home/szu-stu/projects/zrx/code/TreeLearn/work_dirs/ptv3_full_formal_v1_20261023_s42/run
```

不要在已有进程运行时再次执行。停止后恢复仍使用冻结代码、原配置和相同 allocator。原训练器恢复前逐项校验 model、optimizer、scheduler、scaler、RNG 和游标；CUDA 后续轨迹不承诺逐位一致。L1W/Wytham 不参与训练、早停或模型选择，最终模型仍按固定终点选择。

本入口与既有适配器共 **16 项回归测试通过**，日志见 [unit_tests.log](unit_tests.log)，覆盖完整配置/正式计划保护、限点与改通道拒绝、空目录保护、滚动 checkpoint 路由、作用域恢复及已有重计算检查。原 TreeLearn 核心、baseline/vendor、原配置未改；main 和 baseline 分支未动。

实际作业已于 **2026-10-23 16:01（Asia/Shanghai）**启动，PID **1986728**；冻结源 commit 为 **`649c1749ea654cb42add12db2eacff3da9532c09`**。[launch.json](evidence/launch.json) 保存命令与进程信息，[启动状态快照](evidence/startup_run_status.json) 及 [完整性核对](evidence/startup_integrity.json) 记录取证时的真实状态。取证时仍为全量数据 SHA256 核验，正式 update=0；这不是完成训练或通过长期稳定性验证的证据。实时状态以独立运行目录中的 `run/run_status.json` 为准，进程将于全部数据校验通过后自动进入训练。
