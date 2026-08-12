# Q5a：Coverage-Seed Completion Oracle

## 1. 为什么不继续 Q4b1

Q4b1 的三个训练 seed 均未通过：Candidate AP 只有 0.087–0.112，Safety AP
只有 0.196–0.320，最终 validation F1 仅提高 0.061–0.097 pp。K accuracy
约为 90%，说明子树数量不是主要瓶颈；现有实例级特征无法可靠判断“拆谁”和
“是否安全”。因此锁定并关闭实例拆分路线，不调整阈值、不增加注意力，也不读取
Wytham。

Q3 在固定五个 validation forests 上共有 149 棵漏检树：

- base-seed support failure：31；
- undersegmentation：73；
- fragmentation：41；
- semantic support failure：2；
- partial/localization：2；
- density clustering failure：0。

欠分割和安全合并路线已经关闭。31 棵 base-seed support failure 是下一项尚未验证、
且与复杂林分密度和林下遮挡直接相关的错误来源。

## 2. Oracle 定义

GT 只用于指定 Q3 已锁定的 `base_seed_support_failure` 目标树。每棵目标树都具有至少
50 个 semantic tree points，但原始 base seed 数少于 `tau_min=50`。

两个模式都保持 semantic tree mask、预测 base XY votes、HDBSCAN 参数、剩余点
5-NN 分配、checkpoint、tile、ensemble 和官方检测评估定义不变。每棵目标树只补足
到 50 个种子，不增加非 semantic 点：

1. `margin_topup`：用 tree probability、verticality 和 offset-z margin 三个可部署
   预测量排序候选点；
2. `xy_oracle_topup`：在相同候选点中用 GT XY vote error 排序，只作为种子选择能力
   的严格上限，不是可部署方法。

Q5a 会对每个模式重新运行完整 HDBSCAN 和剩余点分配，并重新执行 Hungarian matching，
因此结果会真实反映新增种子造成的恢复、错误合并和 Commission。

## 3. 预注册 Gate

`margin_topup` 要求恢复至少 8/31 棵目标树、F1 至少提高 0.25 pp、最多损失 2 棵
基线已检测树、Commission 增加不超过 0.50 pp，并且至少 4/5 森林 F1 不下降。

`xy_oracle_topup` 要求恢复至少 12/31 棵目标树、F1 至少提高 0.40 pp，其余安全条件
与 margin 模式相同。

决策：margin 通过则进入 Q5b Coverage-Seed Completion Head；仅 XY Oracle 通过则
先重新设计候选种子排序目标；XY Oracle 也失败则关闭 Seed-Completion，不使用 Wytham
调参。

## 4. 运行

```bash
conda activate TreeLearn
mkdir -p logs/coverage_preserving_quality

nohup python -u \
  tools/diagnostics/diagnose_seed_completion_oracle.py \
  --config configs/experiments/coverage_preserving_quality/q5a_seed_completion_oracle.yaml \
  > logs/coverage_preserving_quality/q5a_seed_completion_oracle_run.log \
  2>&1 < /dev/null &

echo $! | tee \
  logs/coverage_preserving_quality/q5a_seed_completion_oracle.pid

tail -f \
  logs/coverage_preserving_quality/q5a_seed_completion_oracle_run.log
```

每个森林会运行一次网络推理和两次额外 HDBSCAN。支持断点续跑；已通过 Q3 一致性检查
的 artifact 会自动 `SKIP`。不要随意使用 `--force`。

查看底层森林进度：

```bash
tail -f logs/coverage_preserving_quality/q5a_seed_completion_oracle/logs/G4N.log
```

判断进程：

```bash
pid=$(cat logs/coverage_preserving_quality/q5a_seed_completion_oracle.pid)
ps -p "$pid" -o pid,%cpu,%mem,rss,etime,stat,cmd
```

完成后：

```bash
cat logs/coverage_preserving_quality/q5a_seed_completion_oracle/summary.md
```

脚本在 Gate 失败时会先保存完整报告，再以 `RuntimeError` 退出；这表示科学 Gate 失败，
不是程序崩溃。
## 5. 固定验证集结果与下一步

Q5a 两个模式均通过。`margin_topup` 恢复 13/31 棵目标树、没有损失基线树，F1
提高 0.511 pp；`xy_oracle_topup` 同样恢复 13 棵但损失 2 棵，F1 提高 0.403 pp。
这说明单点 GT XY vote error 并不是最合适的补种排序目标，预测 margin 的聚类拓扑更好。

下一阶段不是直接训练目标树分类器，而是 Q5b0 GT-free proposal Oracle：先证明仅用
预测 vote-space 多尺度 cell 能覆盖 Q5a 的目标区域，再决定是否训练 Q5b1 激活头。
