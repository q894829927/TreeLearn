# Q3：复杂林分漏检树归因 Oracle

## 为什么停止 Q2

Q2 中 `coverage_cvar` 相对参数匹配 `safe_iou_control` 的 F1 为
`-0.054 pp`，3 个随机种子均未获胜，5 个验证森林只赢 1 个。因此停止
Coverage-CVaR 和实例质量关系注意力路线。另一方面，`safe_iou_control`
的 F1（86.760%）距 Q1 Oracle（87.307%）仅约 0.55 pp，继续优化过滤头
不能解决 Wytham 上主要的漏检问题。

## Q3 目标

使用固定的 G4N、G4W、L1N、O1N、O1W 五个 validation forests，对每棵
GT 树记录以下阶段状态：

1. 语义预测为树的点数；
2. 满足 verticality、offset-z 和语义阈值的 base seed 数；
3. HDBSCAN 初始聚类后仍属于有效簇的 seed 数；
4. 最终预测实例的最佳 IoU、precision、recall 与覆盖率。

漏检树按第一个失败阶段划分为互斥类别：

- `semantic_support_failure`：语义树点少于 `tau_min`；
- `base_seed_support_failure`：语义点足够但 base seed 少于 `tau_min`；
- `density_clustering_failure`：seed 足够但 HDBSCAN 没留下有效 seed；
- `undersegmentation`：存在 recall ≥ 0.5 的实例但未达到 IoU 匹配；
- `fragmentation`：多个片段合计覆盖至少 50%，单片段不足；
- `partial_or_localization`：其余部分覆盖或定位误差。

每个类别同时计算“完美恢复该类全部 FN、且不新增 FP”的检测 F1 上限。
这是诊断 Oracle，不能作为论文方法结果。

## 运行

```bash
conda activate TreeLearn
mkdir -p logs/coverage_preserving_quality

nohup python -u tools/data_gen/gen_gt_omission_diagnostics.py \
  --config configs/experiments/coverage_preserving_quality/q3_omission_bottleneck_oracle.yaml \
  > logs/coverage_preserving_quality/q3_omission_bottleneck_oracle_run.log \
  2>&1 < /dev/null &

echo $! | tee \
  logs/coverage_preserving_quality/q3_omission_bottleneck_oracle.pid

tail -f \
  logs/coverage_preserving_quality/q3_omission_bottleneck_oracle_run.log
```

脚本按森林串行运行。已完成且校验通过的森林会显示 `SKIP`，因此中断后可用
同一命令续跑。只有确实需要重建全部 artifact 时才增加 `--force`。

## 判断进程与查看结果

```bash
pid=$(cat logs/coverage_preserving_quality/q3_omission_bottleneck_oracle.pid)
ps -p "$pid" -o pid,%cpu,%mem,rss,etime,stat,cmd
pgrep -af "[g]en_gt_omission_diagnostics.py"

tail -n 80 \
  logs/coverage_preserving_quality/q3_omission_bottleneck_oracle_run.log

cat data/gt_omission_diagnostics/summary.md
```

当 `ps` 无该 PID、`pgrep` 无输出，且日志末尾已经打印 `PASS` 或明确 traceback，
表示任务已结束。每个森林的底层日志位于：

```text
data/gt_omission_diagnostics/logs/<plot>.log
```

## 固定 Gate

- 五个 validation forests 全部存在；
- GT 类别构成严格分区；
- 总漏检树不少于 20；
- 至少一个类别包含不少于 10 棵漏检树、覆盖不少于 3 个森林，且单类恢复
  Oracle 的 F1 增益不少于 0.30 pp。

Wytham 不得出现在配置或输入路径中。

## Q3 后的唯一下一步

按 `summary.md` 的 `recommended_route` 决策，不自行选择结果最好看的路线：

| 推荐路线 | 下一控制实验 |
|---|---|
| `semantic_recovery` | 冻结实例分组，训练小树/低置信语义恢复头 |
| `seed_recovery` | 冻结主干，做 coverage-aware seed proposal 控制组 |
| `density_aware_grouping` | 不改网络，先做局部密度归一化 HDBSCAN 控制组 |
| `instance_split` | 对低纯度大实例做垂直/拓扑切分 Oracle，再决定网络 |
| `instance_merge` | 关闭此前全局合并，重做仅针对漏检片段的受限合并 Oracle |
| `proposal_localization` | 对 missed-tree votes 做残差/中心定位上限诊断 |

如果 Gate 失败，停止新增模块；不能使用 Wytham 重新选类别或阈值。
