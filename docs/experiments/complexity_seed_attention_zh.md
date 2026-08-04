# 复杂度感知种子注意力实验

## E0：GT Seed Utility Oracle

本阶段只判断“学习种子可靠性是否存在足够上限”，不训练 MLP、Point Transformer 或任何
新模块，也不读取 Wytham。所有模式复用一次 TreeLearn pointwise inference 和 NumPy
ensemble，保证差异只来自初始聚类种子。

固定比较：

1. `baseline`：保留全部基础种子；
2. `random_s42/s43/s44`：随机保留 78%；
3. `oracle_global`：按 GT vote accuracy 与 vote-cell purity 的乘积保留全局最高 78%；
4. `oracle_balanced`：先在每棵 GT 树内部对 utility 排序，再全局保留 78%，模拟后续空间与
   高度平衡选择能够达到的理论上限。

Oracle utility 为：

~~~text
exp(-0.5 * (base_vote_xy_error / 0.30 m)^2)
× vote_cell_purity(0.60 m)
~~~

这是严格的离线上限，使用了 GT offset 与 instance label，不得用于正式 pipeline 或作为方法
结果。只有 `oracle_balanced` 同时提高 F1、降低 Commission、保持 Completeness，并超过三个
随机对照均值，才进入 E1。

## 服务器运行

~~~bash
cd ~/projects/zrx/code/TreeLearn
git switch vertical-instance-quality
git pull --ff-only origin vertical-instance-quality

conda activate TreeLearn
mkdir -p logs/complexity_seed_attention

nohup python -u tools/diagnostics/diagnose_complexity_seed_oracle.py \
  --config configs/experiments/complexity_seed_attention/e0_seed_oracle_l1w.yaml \
  > logs/complexity_seed_attention/e0_seed_oracle_l1w_run.log 2>&1 < /dev/null &

echo $! | tee logs/complexity_seed_attention/e0_seed_oracle_l1w.pid
tail -f logs/complexity_seed_attention/e0_seed_oracle_l1w_run.log
~~~

运行会依次打印一次推理和六次聚类的进度。HDBSCAN 阶段可能数分钟没有新日志，应通过 PID、
CPU 和日志时间共同判断，不要仅凭 `tail -f` 无输出终止。

~~~bash
pid=$(cat logs/complexity_seed_attention/e0_seed_oracle_l1w.pid)
ps -p "$pid" -o pid,%cpu,%mem,rss,etime,stat,cmd
cat logs/complexity_seed_attention/e0_seed_oracle_l1w/summary.md
~~~

## E0 Gate

- F1 相对 baseline 至少提高 `0.50 pp`；
- Commission 至少降低 `1.00 pp`；
- Completeness 下降不超过 `0.50 pp`；
- F1 至少高于三个 random controls 的均值 `0.25 pp`。

若脚本最后抛出 `Seed Oracle gate failed`，表示诊断正常完成但上限不足，应停止本路线；不得
通过修改 Wytham 参数寻找正结果。若 PASS，E1 才实现完全相同输入与标签的 MLP 控制组，随后
实现单层 Point Transformer；Attention 必须相对 MLP 提高 seed-ranking 指标后才能进入完整
pipeline 实验。
