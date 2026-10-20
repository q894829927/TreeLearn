# ForestABC 20000 步 Wytham 结果与已有预测诊断

本次数据来自用户在服务器运行后提供的日志；开发机没有执行模型、推理或测试。

## 实测结论

| 指标 | 官方权重基线 | ABC 20000 步 |
| --- | ---: | ---: |
| 五样地验证 F1 | 88.19% | 92.11% |
| 五样地验证 Coverage | 82.08% | 83.70% |
| Wytham F1 | 70.32% | 44.88% |
| Wytham 检出率 | 67.96% | 35.01% |
| Wytham 误检率 | 27.14% | 37.47% |
| Wytham Coverage | 65.13% | 37.35% |
| Wytham 点级精确率 | 71.23% | 41.12% |
| Wytham 点级召回率 | 80.37% | 79.34% |

同域验证提升没有迁移到这份 Wytham。高点级召回与低精确率、低实例检出率的组合，优先提示实例合并、邻树或背景混入；仅凭平均指标不能确认具体原因，更不能直接归因于某个模块。

Wytham 预测完整处理了 500 个块，没有跳块，490 个块触发了 98,304 体素上限的抽样。150,000 个 query 经质量筛选和局部 NMS 后保留 12,008 个提案；全局 NMS/分配后输出 1,049 个实例，2,039 个候选未通过分配验收。原流程推理约 68.8 分钟。

877 棵 GT 中，ABC 检出 307 棵，漏检 570 棵，184 个预测计为 FP。其余 558 个最终实例未计入 TP/FP，可能完全位于负标签忽略区，或有效点中的 GT 树点占比不足 50%；需要逐实例计数区分，不能直接把它们算作误检或额外正确树。候选数、最终实例数、TP 数是不同计数。

抽样频繁值得检查，但当前记录没有提供抽样前后或局部提案质量的对照，不能据此认定抽样造成此次失败。与训练数据的实例尺度差异见[数据分析](forest_abc_data_driven_next_steps_20261020_zh.md)。

## 不重新推理的 CPU 诊断

[diagnose.py](../../tools/forest_abc/diagnose.py) 仅读取 `coords.npy`、`labels.npy`、`baseline.npy` 和已有 `refined_labels.npy`。用两遍分块扫描构建计数和 GT 竖向跨度，不加载 torch、spconv、模型或 checkpoint，不修改预测或阈值。

它先复算两套预测的全部指标，并与已有 `metrics.json` 核对；文件错配、局部预测、实例计数不一致或扫描中输入文件改变时拒绝生成成功报告。文件时间/大小与指标一致性校验不等同于独立证明全部数据来源，评估目录仍须由用户正确指定。

前台执行，锁与训练共用；不要同时改写该评估目录：

```bash
cd ~/projects/zrx/code/TreeLearn
conda activate TreeLearn
git pull --ff-only origin codex/candidate-mask-mlp-step1
python -m unittest discover -s tests -p 'test_forest_abc_diagnostics.py' -v
set -o pipefail
flock -n logs/forest_abc_v1/workflow.lock \
  python -u tools/forest_abc/diagnose.py \
  --split external --plot wytham_vox0.1 --expect-step 20000 \
  2>&1 | tee logs/forest_abc_v1/diagnose_wytham_step_020000.log
```

结果目录为 `work_dirs/forest_abc_v1_s42/diagnostics/external_step_020000/wytham_vox0.1/`：

- `brief.json`：基线和 ABC 指标、四类最终实例计数、GT 错误类别、基线检出而 ABC 丢失/反向恢复的棵数、按 GT 竖向跨度分组的结果。
- `trees.csv`：逐 GT 的点数、竖向跨度、最佳预测 IoU/精确率/召回率、分给任意实例的点比例、最佳预测的邻树点和背景点比例，以及是否有一个预测覆盖多棵 GT 各至少一半的点。
- `baseline_predictions.csv`、`refined_predictions.csv`：逐预测的匹配类别、已知/忽略点、GT 树点比例、主导 GT 与覆盖 GT 数。
- `summary.json`：以上汇总加输入文件来源记录、原推理统计和解释限制。

GT 错误类别互斥：`detected` 为成功检出；`no_assigned_tree_points` 为整棵树无点分到实例；`high_recall_low_purity` 为最佳 IoU 预测召回至少一半、纯度不超过一半；`spread_across_predictions` 为至少一半 GT 点被分配、但没有任何单个实例覆盖一半；`mostly_unassigned` 为不足一半 GT 点被分配；其余为 `other_below_iou_threshold`，另单列具有合格 IoU 却未匹配的异常情况。

`z_span_m=max(z)-min(z)` 只是标注实例的竖向跨度，不是地面归一化树高。分组只比较 GT 检出率和分割指标，不把背景预测强行归属某个高度组，不报告分组 F1。低纯度可能来自邻树或背景，CSV 分开列出两者；一个预测覆盖多棵 GT 各一半是最终掩码合并的证据，不能确定其上游原因。

这些诊断只解释最终分配后的掩码。原始提案已按默认配置删除，不能从这些文件恢复 query 覆盖、NMS 前的掩码质量，或证明被丢弃的候选原本正确。如需逐阶段归因，再有针对性地增加推理记录。

此提交只新增独立工具、服务器 CPU 回归用例和文档，不改训练/模型/配置/已有评估实现，不改变训练恢复签名。提交前只做静态检查；用例留在服务器执行。
