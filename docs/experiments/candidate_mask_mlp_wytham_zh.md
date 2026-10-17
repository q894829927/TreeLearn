# 固定 A1 checkpoint 的 Wytham 跨域评价

更新日期：2026-10-17。用户明确提出在 Wytham 上测试，源域 gate 失败不作为禁止本次外部评价的条件。流式切块修复后，用户已回传最终评价：F1 70.324% → 66.667%，Coverage 65.125% → 61.911%，当前 A1 在 Wytham 上也退化。完整数字与判定见 [结果记录](candidate_mask_mlp_wytham_results_20261017_zh.md)。以下命令用于复现，无需重复执行已完成的实验；本地未运行点云实验或测试。

## 1. 为什么仍值得测试

源域高分可能压缩改进空间，源域退化也不能推导出目标域一定退化。仓库历史结果中官方 TreeLearn 在 Wytham 的 F1 约 72%、Coverage 约 57.7%，说明该场景更有挑战；这些旧值来自既有 Wytham 评价，不与本轮 aligned-cache 指标直接相减。

此前“源域失败就不测 Wytham”的安排过于严格。此次检验独立的问题：**用源域选定并冻结的 A1，是否在 Wytham 上比同设置的原版预测更好？** 不能靠这次结果覆盖原验证失败，也不能在看到分数后重新选 checkpoint 或阈值。

Wytham 已被项目早期实验观察，因此本轮明确记为探索性跨域评价，不声称它是全流程未见的最终测试集。这个定位不妨碍测试本身的价值。

## 2. 固定什么

- 模型：源域 `work_dirs/candidate_mask_mlp_a1_s42/best.pth`。此前 history 最佳验证损失对应第 26 轮，脚本将从实际 checkpoint 记录 epoch。
- 主干：训练缓存对应的官方权重、预处理和分组配置，外部缓存的 baseline signature 必须与训练缓存一致。
- MLP：模型参数和源域训练得到的 feature_mean/feature_scale 原样加载，不在 Wytham 重估归一化统计。
- 候选区域、成员阈值 0.5、分配规则：沿用源域设置。
- A0/A1：在同一份新生成的 Wytham 缓存上配对评价，A0 是原单 anchor 分组后的完整预测。

外部脚本只有 `prepare`、`evaluate` 两个命令，没有训练或选 checkpoint 的入口。不要把 Wytham 加进 `step1.yaml` 的训练/验证森林列表。

首次 `prepare` 写入独立的 `external_lock.json`，记录 checkpoint SHA256、epoch、原验证结果摘要 SHA256 和输入路径。后续 checkpoint 或设置变化会拒绝混用本次结果。原 `passed=false` 写进外部报告，不阻止运行；外部评价不套用五森林源域 gate，`passed` 为 null，直接报告配对指标及代价。

默认外部配置现在启用 `stream_tiles: true`：逐块生成和保存，避免同时保留全部重叠块。通过单独的 `pipeline_streaming.py` 入口替换当前进程的切块方法，继续调用原版流水线。原版六个 baseline 源文件、模型配置及训练 manifest 不变；新的实现代码哈希另记在外部锁、缓存签名/元数据、manifest 和最终报告的 `preparation` 中，不把实现变更隐去。只支持当前基线的无 plot rotation、无去噪设置，其他设置会明确拒绝。

原先因 OOM 失败、尚未生成 `cache.json`/manifest 的尝试可以在保留原权重与数据锁的基础上首次绑定 streaming 实现；已完成的旧缓存不能直接改用新实现，已绑定的实现哈希变化也会拒绝混用。当时两次失败尚未写出完整缓存，之后已沿用原目录恢复并完成最终评价。

## 3. 服务器命令

默认数据路径为仓库既有配置使用的 `data/benchmark/wytham_vox0.1.laz`。若服务器保存位置不同，只修改 [外部配置](../../configs/experiments/candidate_mask_mlp/wytham.yaml) 的 `forest_path`，指向已有的同一份 0.1 m 带标签 Wytham 点云。不要误用 4.64 亿点的原始云。读取原文件，生成内容写入独立目录。

从仓库根目录执行；每一步成功后再运行下一步：

```bash
cd ~/projects/zrx/code/TreeLearn
git switch codex/candidate-mask-mlp-step1
git pull --ff-only
conda activate TreeLearn
mkdir -p logs/candidate_mask_mlp
set -o pipefail

python -m unittest discover -s tests -p 'test_candidate_mask_mlp.py' -v \
  2>&1 | tee logs/candidate_mask_mlp/unit_tests_external_streaming.log
```

应发现 34 项测试。新增检查覆盖：失败缓存首次绑定 streaming 与已完成缓存的保护、与实际原版方法逐项相同的 CPU/CUDA tile 输出、保存后释放块视图、拒绝尚未验证的旋转/去噪。CPU 比较覆盖 float32/float64、两种特征维度、两种压缩模式、两种 stride，以及边界点和相邻浮点数；CUDA 可用时还会直接比较两实现的 GPU 结果，无 CUDA 时该项跳过。先确认这组测试通过，再启动实际 Wytham prepare。

长任务使用后台脚本启动，减少 VS Code 或 SSH 断开造成的影响。它继承当前已激活的 Python 环境，先执行 `conda activate TreeLearn`。下面的 `prepare` 启动后会立即返回终端：

```bash
nohup bash tools/mask_refinement/run_external.sh prepare \
  > logs/candidate_mask_mlp/launch_prepare.log 2>&1 < /dev/null &
```

查看本次启动情况及两层运行日志：

```bash
cat logs/candidate_mask_mlp/launch_prepare.log
tail -n 30 logs/candidate_mask_mlp/background/latest_prepare/console.log
tail -n 30 data/candidate_mask_mlp_a1_wytham_s42/logs/wytham_vox0.1.log
```

脚本使用 `flock` 拒绝重复后台运行，并在启动前检查当前用户是否已有 `external.py` 或 `pipeline.py` 进程；不会停止现有进程。若仅有 `launch_prepare.log` 而没有新运行目录，先读启动错误。服务器需有 `flock`、`pgrep`、GNU `/usr/bin/time` 和 `/proc`。每次运行产生独立目录，`latest_prepare`/`latest_evaluate` 指向最新一次；不要在运行中手动改动这些软链接。

只有 prepare 的退出码为 0，并且 console 日志出现 `LOCKED external checkpoint ...`，才继续 evaluate。运行期间尚无退出码文件时，`cat` 会提示不存在：

```bash
cat logs/candidate_mask_mlp/background/latest_prepare/exit_code.txt

# 确认上一条输出为 0 后，单独执行：
nohup bash tools/mask_refinement/run_external.sh evaluate \
  > logs/candidate_mask_mlp/launch_evaluate.log 2>&1 < /dev/null &
```

评价结束后检查 `background/latest_evaluate/console.log` 和 `exit_code.txt`。也可以在持久终端中直接执行 `python -u tools/mask_refinement/external.py prepare`/`evaluate`；后台脚本仅增加运行记录，不改变实验参数。

`prepare` 需要 GPU 运行一次冻结 TreeLearn，生成 32 维特征和原版预测，因此不能直接用旧 LAZ 预测代替此缓存。它会单独生成 tiles，不改旧 Wytham 实验的 tiles/results。准备阶段较耗时，森林细节日志位于 `data/candidate_mask_mlp_a1_wytham_s42/logs/wytham_vox0.1.log`。区域超过现有大小限制会停止，不自动缩小区域或截断点。

`evaluate` 只加载已选定的 MLP 推理，不重新训练。Wytham GT 在掩码预测返回后用于算指标，不用于选择模型或归属。需要可视化点云时可追加 `--export`；否则仍保存 refined_labels 和 unassigned_tree_mask。

### 2026-10-16 中断排查与恢复

服务器回传日志确认：28 项单元测试通过，prepare 完成体素化、特征计算和切块筛选，最后停在 `21:07:38 center chunks`；没有 `save chunks`、Traceback、评价日志或最终报告，检查时也没有对应进程。1392 个 NaN 特征已成功替换，不能作为此次退出的证据。

随后服务器显示物理内存 125 GiB、当时可用约 100 GiB，swap 2 GiB 接近用完，开机时间为 2026-08-19。仅凭这次事后快照不能判断峰值，普通用户也无系统内核日志权限，因此当时尚不能确认原因。

恢复时拉取代码、激活环境，然后使用上述后台命令重新执行 prepare；不删除输入、源域 checkpoint 或外部锁文件。已完成的体素化和特征文件由原流水线复用；切块阶段会重新执行。每次重新启动流水线，已有森林日志自动改名归档为 `wytham_vox0.1.<时间戳>.log`。此次更改只涉及启动和日志，训练缓存绑定的六个 baseline 源文件及配置保持原样，原 checkpoint 的签名继续适用。

每次后台运行保留：

- `console.log`：Python 环境、命令、流水线退出提示和外部脚本异常栈。
- `resources.log`：每 10 秒记录主机可用内存、swap 及当前用户内存占用最高的进程（RSS 单位 KiB，不记录命令参数）。若服务器开放 cgroup v2 对应文件，同时记录所在组的 `memory.current`、`memory.max` 和 `memory.events`，用于判断是否存在小于整机内存的作业限制及 OOM 计数变化；同组其他进程也可能贡献这些计数。短暂峰值可能落在采样间隔内。
- `time.txt`：GNU time 的耗时、最大 RSS、退出信息；子流水线被信号终止时，console/森林日志额外记录 `SIGKILL` 等信号。
- `runner.pid`、`task.pid`、`exit_code.txt`：启动进程信息和最终状态。退出码文件缺失需要结合进程和日志判断；整个进程组被强制杀死时可能来不及写结束信息。`SIGKILL` 也可能来自人工或调度器，不能单独认定 OOM。

异常时回传这些记录的末尾：

```bash
tail -n 40 logs/candidate_mask_mlp/background/latest_prepare/console.log
tail -n 60 logs/candidate_mask_mlp/background/latest_prepare/resources.log
cat logs/candidate_mask_mlp/background/latest_prepare/time.txt
cat logs/candidate_mask_mlp/background/latest_prepare/exit_code.txt
tail -n 40 data/candidate_mask_mlp_a1_wytham_s42/logs/wytham_vox0.1.log
```

若后台运行仍被服务器会话清理策略终止，或内核证据仍不可见，再向管理员索取对应时间段的 OOM/作业终止记录。

### 2026-10-17 确认 OOM，改为逐块保存

用户回传第二次后台运行（服务器时间 2026-10-16 21:24:29–21:26:36）的资源记录：

- 21:26:19：流水线 Python RSS 为 86,186,116 KiB，主机可用内存 19,184,448 KiB。
- 21:26:29：RSS 升至 101,760,304 KiB，主机可用仅 3,705,820 KiB，swap 已用完。
- 21:26:36：流水线收到 SIGKILL，cgroup `oom_kill` 从 0 增至 1，可用内存恢复为 105,703,288 KiB。GNU time 记录最大 RSS 105,469,508 KiB，约 100.6 GiB。该组 `memory.max=max`，这份记录未显示该组的独立硬上限。

这次可以确认 OOM。原实现把所有重叠块先放入 `chunks`/`valid_chunks`，随后中心化所有块才开始保存；中心化还会因减去 float64 center 而将块提升到 float64。新实现保持相同 grid、边界不等式、float32 bounds 舍入、CUDA 中心化算术、float32 输出和编号顺序，每次处理一个有效块即保存并释放其数组/视图。内存不再随所有重叠块点数的总和增长，但森林原始数组和单个块仍需内存；实际峰值由下一次服务器资源记录确认。

不缩小 tile、不改变 stride/体素大小、不截断点，也不重新训练。先拉取修复、运行 34 项测试，再按前述 nohup 命令重试；旧日志继续自动归档，已有体素化和特征文件会复用。成功日志应包含 `streaming tiles: saved ...`、`streaming tiles complete ...`，随后进入冻结主干的推理流程。

## 4. 输出与解释

回传 `work_dirs/candidate_mask_mlp_a1_wytham_s42/evaluation/summary.json`。它记录实际 checkpoint/epoch/哈希、原源域是否通过、Wytham A0/A1 的 F1、Coverage、Completeness、Commission、点精度/召回、未分配点与候选消失，以及恢复/丢失的 GT。

协议仍为 `aligned_voxels_ignore_negative_gt_iou_gt_0.5`，沿用本轮 A1 定义。旧 Wytham 日志经过预测到 GT 点云的传播等流程，不直接把旧 F1 当作本次 A0；先看同一个新 summary 内 A1 相对 A0 的变化。

如果 Wytham 改善而源域退化，应报告跨域收益及源域代价，不能写成全面改善；如果两边都退化，则更有理由结束当前 A1 版本。若后续根据 Wytham 结果选择方法，应把它明确作为开发数据，并另设未参与选择的评价数据。这次不实现新的 A1b 或 attention。

本次回传结果为两边均退化，结束当前版本的追加试验。外部 `passed=null` 表示没有套用五森林源域 gate，不表示本次成功；无需重新训练或重跑 Wytham 来改变该结论。
