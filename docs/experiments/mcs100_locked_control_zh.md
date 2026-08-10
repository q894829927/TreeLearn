# E3a-Control：锁定 mcs=100 全局分组实验

## 定位

E3a 多尺度 proposal Oracle 的主 Gate 已失败，因此多尺度选择器与复杂度感知分组路线
保持关闭。E3a 的固定验证网格同时显示：单一 `min_cluster_size=100` 相对原版 `50`：

- Tree recall：96.698% → 96.632%，下降 0.066 个百分点；
- Commission：30.199% → 24.930%，下降 5.269 个百分点；
- seed-cluster F1：79.577% → 83.488%，提升 3.911 个百分点。

因此增加一个普通的全局超参数控制实验。它不属于新模块，也不能被表述为注意力或
复杂度自适应方法。`100` 已依据固定 validation forests 锁定，Wytham 只用于一次
development 复核，不得继续搜索 75、125、150 等数值。

## 运行顺序

必须先运行 L1W。L1W pipeline/evaluation 正常完成后，才运行一次 Wytham。

```bash
mkdir -p logs/complexity_seed_attention

nohup bash -c '
set -e

echo "===== START L1W mcs100 pipeline $(date) ====="
python -u tools/pipeline/pipeline.py \
  --config configs/experiments/complexity_seed_attention/pipeline_l1w_mcs100_locked.yaml \
  > logs/complexity_seed_attention/mcs100_pipeline_l1w.log 2>&1

echo "===== START L1W mcs100 evaluation $(date) ====="
python -u tools/evaluation/evaluate.py \
  --config configs/experiments/complexity_seed_attention/evaluate_l1w_mcs100_locked.yaml \
  > logs/complexity_seed_attention/mcs100_evaluate_l1w.log 2>&1

echo "===== START Wytham mcs100 pipeline $(date) ====="
python -u tools/pipeline/pipeline.py \
  --config configs/experiments/complexity_seed_attention/pipeline_wytham_mcs100_development.yaml \
  > logs/complexity_seed_attention/mcs100_pipeline_wytham.log 2>&1

echo "===== START Wytham mcs100 evaluation $(date) ====="
python -u tools/evaluation/evaluate.py \
  --config configs/experiments/complexity_seed_attention/evaluate_wytham_mcs100_development.yaml \
  > logs/complexity_seed_attention/mcs100_evaluate_wytham.log 2>&1

echo "===== FINISHED mcs100 locked control $(date) ====="
' > logs/complexity_seed_attention/mcs100_locked_runner.log 2>&1 < /dev/null &

echo $! | tee logs/complexity_seed_attention/mcs100_locked_runner.pid
tail -f logs/complexity_seed_attention/mcs100_locked_runner.log
```

查看阶段进度：

```bash
tail -n 40 logs/complexity_seed_attention/mcs100_locked_runner.log
tail -n 30 logs/complexity_seed_attention/mcs100_pipeline_l1w.log
tail -n 30 logs/complexity_seed_attention/mcs100_evaluate_l1w.log
tail -n 30 logs/complexity_seed_attention/mcs100_pipeline_wytham.log
tail -n 30 logs/complexity_seed_attention/mcs100_evaluate_wytham.log
```

汇总指标：

```bash
for plot in l1w wytham; do
  echo "========== ${plot} mcs100 =========="
  grep -E \
    "Completeness:|Commission Error Rate:|F1 Score:|Precision:|Recall:|Coverage:" \
    logs/complexity_seed_attention/mcs100_evaluate_${plot}.log
done
```

## 判定

Wytham 对照为：F1 72.081%、Completeness 64.766%、Commission 18.741%、
Coverage 57.718%。

- 若 mcs100 的 Wytham F1 至少提高 0.5 个百分点，且 Completeness 下降不超过
  1.0 个百分点：保留为更强的调参基线，随后重新评价质量排序是否仍提供互补收益。
- 否则：仍使用 mcs50 原版基线，mcs100 只放入参数敏感性附录。

无论结果如何，mcs100 都不能单独作为论文创新点。
