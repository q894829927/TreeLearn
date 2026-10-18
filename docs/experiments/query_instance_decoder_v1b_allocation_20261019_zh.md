# 查询解码器 v1b：分配后质量校正

日期：2026-10-19。依据用户回传的 CPU 诊断结果，固定使用 v1 第 2000 step 权重，增加分配后整实例过滤。未重新训练、未更改模型参数或损失。源域结果是同一开发集上的事后改进，不是独立测试，也不把收益单独归因于 attention。

## 1. 本次学到了什么

v1 的 1151 个计数 FP 中，891 个分配后剩余点数不到原掩码一半，占约 77.4%；907 个 FP 的最佳重叠 GT 已由其他 TP 匹配，占约 78.8%。两组可能重叠，不能相加，也不能仅据最佳 GT 相同认定全部是重复实例。232 个 FP 的原质量分数仍至少为 0.7，说明只提高原分数阈值不能充分解决问题。

| 方法 | macro F1 | TP | FP | FN |
|---|---:|---:|---:|---:|
| 原版 TreeLearn | 88.195% | 1633 | 275 | 149 |
| v1 原始输出 | 69.531% | 1608 | 1151 | 174 |
| 仅保留分配后至少一半点 | 87.385% | 1604 | 260 | 178 |
| 仅原质量分数至少 0.6 | 82.505% | 1589 | 436 | 193 |
| 原质量 × 保留比例至少 0.4 | **89.831%** | **1601** | **165** | **181** |

组合规则相对 v1 去掉 986 个 FP、损失 7 个 TP，F1 提升 20.300 pp；相对原版 F1 +1.636 pp、Commission -4.642 pp。单独残片过滤已能恢复大部分 F1 退化，支持“分配后残片仍沿用原分数”是本轮重要问题的判断。

收益带有召回取舍：相对原版 Completeness -1.652 pp、Coverage -1.504 pp、Point recall -2.843 pp，不能称全面超过原版。它没有恢复点或新匹配树，仅删除实例后取得更好的精确率/F1。下一阶段应关注保留更多有效树和树点，而不是继续扩大这套冻结 v1 的训练步数。

## 2. 固定实现规则

```text
retained_fraction = 分配后点数 / 原掩码点数
allocation_score = 原质量分数 × retained_fraction
保留 allocation_score >= 0.4 的实例
```

实现先完成原来的全部 NMS 和按质量排序的互斥分配，再根据该分数删除整个实例，删除点变为 0，不给其他实例重新占用。这样与本次诊断的固定最终掩码过滤一致。提前拒绝掩码并释放点给后续查询会改变后续掩码，不是这次已验证的规则。

新标签将保留实例重新编号为 1..K，`instances.json` 记录 `source_instance_id`、原 `score`、`retained_fraction` 和校正后的 `allocation_score`，可追踪原实例。过滤函数只接收标签、预测分数及点数，不读取 GT；GT 仅在输出完成后用于评价。

原 v1 的模型权重、原始评价和默认未过滤路径保留。新结果写入 `evaluation_allocation_filtered/`，没有把 v1 原始 F1 改写成 89.83%。新增 4 项服务器测试覆盖诊断与实际标签一致、分 chunk 与 ID 映射、空输出/错误点数、原生预测与缓存过滤一致、输入文件保留。本地只执行 Python AST、Bash 语法、文档及 Git 静态检查，测试和新导出留给服务器。

## 3. 当前先执行：CPU 导出可用标签

```bash
cd ~/projects/zrx/code/TreeLearn
git pull --ff-only
conda activate TreeLearn
bash tools/query_segmentation/launch.sh filter
```

按用户偏好，启动器前台先执行 4 项测试，再读取现有五森林预测，导出新标签并计算完整评价；当前终端实时显示全部阶段进度，同时保存日志。无需重新跑 GPU、生成缓存或训练。当已有诊断与当前原始评价的哈希一致时，会逐森林核对导出指标与诊断是否一致。

```bash
cat logs/query_instance_decoder_v1/latest/status.txt
tail -n 20 logs/query_instance_decoder_v1/latest/filter.log
cat work_dirs/query_instance_decoder_v1_s42/evaluation_allocation_filtered/brief.json
```

完整指标在同目录 `summary.json`，每片森林输出 `refined_labels.npy`、`instances.json`、`rejected_instances.json`、`allocation_filter.json`。实际导出预期复现诊断 F1 89.8309%；当前尚待服务器执行。`diagnostic_reference_verified=true` 表示导出与对应诊断逐森林核对成功。

## 4. 再执行：固定规则的 Wytham 评价

等待 `filter` 的状态为 `DONE` 后，可以运行：

```bash
bash tools/query_segmentation/launch.sh external-filtered
```

它复用已有 Wytham 的 TreeLearn 特征缓存，以固定的源域第 2000 step 权重执行查询解码器，并使用同一条分配后过滤规则，不用 Wytham GT 生成预测或修改阈值。无需训练或重新生成原始 2009 个 tiles。Wytham 先前参与过开发分析，这仍是探索性跨域验证。

```bash
cat logs/query_instance_decoder_v1/latest/status.txt
tail -n 20 logs/query_instance_decoder_v1/latest/external.log
cat work_dirs/query_instance_decoder_v1_s42/external_evaluation_allocation_filtered/summary.json
```

若要从头重做源域查询预测以检查同一规则，可用 `bash tools/query_segmentation/launch.sh evaluate-filtered`；它需要 GPU，当前 CPU 导出足够时无需重复。三个入口共用任务锁，应先后执行。

CLI 的对应原生预测开关为 `python -u tools/query_segmentation/run.py evaluate --allocation-filter`；外部评价再加 `--external`。结果目录按规则区分，输出记录 checkpoint、代码哈希和规则来自开发集选择的事实。源域同一 best 权重是否在新规则下仍为全程最佳，本轮没有重新扫描 checkpoint，后续有必要时另行比较。

诊断数字精简存档：[JSON](query_instance_decoder_v1b_diagnosis_20261019.json)。v1 原始负结果：[完整记录](query_instance_decoder_v1_results_20261019_zh.md)。
