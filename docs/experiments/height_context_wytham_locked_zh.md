# HSCA Wytham 锁定跨域实验

## 数据角色

L1W 已接近饱和，因此本实验把 Wytham 作为复杂林分跨域开发基准。Wytham 已在
前期研究中使用过，不能描述为完全未见的独立测试集。本轮只允许一次性比较锁定的
A0/A1/A2，不得根据结果重新选择 epoch、阈值、聚类参数或网络结构。

## 对照组

- A0：官方 TreeLearn checkpoint；
- A1：冻结主干的参数匹配 Height-MLP；
- A2：冻结主干的 Height-Stratified Context Attention；
- 三组统一使用 legacy base anchor、base-only 2D grouping、`tau_min=50`、全部种子；
- 三组统一使用 `voxelized_and_filtered` 输出和官方 Wytham evaluation protocol。

## 配置预检

```bash
cd ~/projects/zrx/code/TreeLearn
conda activate TreeLearn
mkdir -p logs/height_context_attention

python tools/diagnostics/verify_height_context_wytham_configs.py \
  2>&1 | tee logs/height_context_attention/verify_wytham_configs.log
```

只有输出以下内容才继续：

```text
PASS: Wytham A0/A1/A2 configs are locked and directly comparable.
```

## 一次性运行

```bash
nohup bash -c '
set -e

for experiment in a0_official a1_height_mlp a2_hsca; do
  echo "===== START Wytham pipeline ${experiment} $(date) ====="
  python -u tools/pipeline/pipeline.py \
    --config "configs/experiments/height_context_attention/pipeline_wytham_${experiment}_locked.yaml" \
    > "logs/height_context_attention/pipeline_wytham_${experiment}_locked.log" 2>&1

  echo "===== START Wytham evaluation ${experiment} $(date) ====="
  python -u tools/evaluation/evaluate.py \
    --config "configs/experiments/height_context_attention/evaluate_wytham_${experiment}_locked.yaml" \
    > "logs/height_context_attention/evaluate_wytham_${experiment}_locked.log" 2>&1

  echo "===== FINISHED Wytham ${experiment} $(date) ====="
done
' > logs/height_context_attention/wytham_a0_a1_a2_locked_runner.log 2>&1 < /dev/null &

echo $! | tee logs/height_context_attention/wytham_a0_a1_a2_locked_runner.pid
tail -f logs/height_context_attention/wytham_a0_a1_a2_locked_runner.log
```

## 汇总

```bash
for experiment in a0_official a1_height_mlp a2_hsca; do
  echo "========== Wytham ${experiment} =========="
  grep -E \
    "Completeness:|Commission Error Rate:|F1 Score:|Precision:|Recall:|Coverage:" \
    "logs/height_context_attention/evaluate_wytham_${experiment}_locked.log"
done
```

## 预先锁定的判断

- A2 相对 A0 Detection F1 至少提升 `1.0 pp`；
- A2 相对 A1 Detection F1 至少提升 `0.5 pp`；
- 或 A2 相对两者 Coverage 至少提升 `1.0 pp`，且 Precision 下降不超过 `1.0 pp`；
- A2 不得出现明显 Completeness 下降。

只有 A2 同时超过 A0 和参数匹配 A1，才能把注意力作为正创新点。若 A1 超过 A0、
但 A2 不超过 A1，则保留高度适配器，HSCA 作为负消融。三组结果产生后不得继续在
Wytham 上修改参数。
