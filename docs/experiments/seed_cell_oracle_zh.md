# E1d Seed-Cell 粒度 Oracle 实验

## 目的

E1c 的参数匹配 Relation-Attention 未优于 Neighborhood-MLP：Reliability AUC
下降 0.001154，tree-only AUC 下降 0.000990，selected utility 下降
0.001108，仅 Critical recall 提高 0.004495。因此不得继续扫描逐点邻域半径、K、层数或
降低 Gate。

E1d 不训练新网络，而是判断建模粒度是否错误。TreeLearn 的最终实例来自 base-vote 空间的
密度聚类，单个种子的标签不能完整描述其对簇连通性和密度拓扑的影响。因此将候选种子按
base-vote XY 聚合为固定 0.60 m Seed Cells，并测试 cell-balanced GT Oracle 是否同时保留
逐点 Oracle 的质量、改善密度平衡并稳定优于同配额随机对照。

本阶段只读取 E1a 固定的五个 validation forests：G4N、G4W、L1N、O1N、O1W。严禁读取
Wytham、重新搜索 cell size、keep ratio 或 density exponent。

## 固定协议

- vote cell size：0.60 m；
- 候选保留比例：0.78，使用 `ceil(N × 0.78)` 精确预算；
- cell 配额权重：`sqrt(cell candidate count)`，即 density exponent=0.50；
- 所有 `target_coverage_critical` 候选必须保留；
- 每个占用 cell 至少保留一个候选；
- cell 达到容量后，剩余预算重新分配给未饱和 cell；
- 随机对照固定 seed 42、43、44，并共享相同 cell 配额与 mandatory 集合。

比较模式：

1. `point_oracle`：mandatory seeds + 全局 GT utility 排序；
2. `cell_oracle`：固定平方根密度配额，在每个 cell 内按 GT utility 排序；
3. `cell_random_s42/s43/s44`：相同配额，在每个 cell 内随机排序。

这里的 GT utility、coverage-critical 标签只用于估计表示上限，不能进入正式 pipeline。

## 服务器同步与测试

~~~bash
cd ~/projects/zrx/code/TreeLearn
git switch vertical-instance-quality
git pull --ff-only origin vertical-instance-quality

conda activate TreeLearn
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export PYTHONUTF8=1
mkdir -p logs/complexity_seed_attention

python -m unittest \
  tests.test_seed_cell_oracle \
  tests.test_seed_cell_oracle_integration \
  -v
~~~

必须显示8项测试全部 `OK`。

## 运行 E1d

本阶段只处理已经生成的 E1a NPZ，不调用 GPU，不重新运行 TreeLearn pipeline。通常几分钟内
完成，可使用前台命令，便于立即发现 artifact 路径或数据完整性问题：

~~~bash
set -o pipefail

python -u tools/diagnostics/diagnose_seed_cell_oracle.py \
  --config configs/experiments/complexity_seed_attention/e1d_seed_cell_oracle.yaml \
  2>&1 | tee logs/complexity_seed_attention/e1d_seed_cell_oracle_run.log
~~~

如果希望后台运行：

~~~bash
nohup python -u tools/diagnostics/diagnose_seed_cell_oracle.py \
  --config configs/experiments/complexity_seed_attention/e1d_seed_cell_oracle.yaml \
  > logs/complexity_seed_attention/e1d_seed_cell_oracle_run.log \
  2>&1 < /dev/null &

echo $! | tee logs/complexity_seed_attention/e1d_seed_cell_oracle.pid
tail -f logs/complexity_seed_attention/e1d_seed_cell_oracle_run.log
~~~

完成后查看：

~~~bash
cat logs/complexity_seed_attention/e1d_seed_cell_oracle/summary.md

column -s, -t < \
  logs/complexity_seed_attention/e1d_seed_cell_oracle/per_plot_metrics.csv | \
  less -S
~~~

末尾抛出 `E1d Seed-Cell Oracle gate failed` 表示诊断正常完成但上限不足，不是程序崩溃。

## 预注册 Gate

E1d 必须同时满足：

- 五个固定 validation forests 全部存在；
- 总 Seed Cells 不少于 10,000；
- cell identity 解释的 utility 方差比例不少于 5%；
- Cell Oracle selected utility 相对 cell-random 均值提高至少 3%；
- Cell Oracle non-tree rate 相对 cell-random 均值降低至少 20%；
- Cell Oracle 保留 point Oracle 至少 97% 的 selected utility；
- Critical recall 相对 point Oracle 最多下降 0.5 个百分点；
- 每个森林 Critical-tree coverage 至少 99%；
- selected cell-count CV 相对 point Oracle 降低至少 5%；
- 至少 4/5 森林同时取得 utility 提升、non-tree rate 降低与99%树覆盖。

## 下一阶段

- PASS：实现同一 Cell Token 输入、同三随机种子、参数匹配的 Seed-Cell MLP 控制组与
  Superpoint Query Adapter。新模块预测 Offset-XY residual、Reliability 和 Coverage；
  residual 零初始化，只解冻最后一个 U-Net decoder block与新模块。
- FAIL：Seed-Cell 粒度也没有足够上限，正式关闭种子注意力路线。不得在 Wytham 上搜索
  cell size 或密度指数，回到已验证的实例质量排序与选择性风险控制路线。
