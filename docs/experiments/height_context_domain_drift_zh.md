# D1 HSCA 跨域预测漂移诊断

## 目标

锁定的 Wytham 对照显示，Height-MLP 与 HSCA 都降低了 Commission，但同时降低了
Completeness、Recall 和 Coverage。D1 不读取 GT，也不选择任何新参数，只回答冻结
适配器相对官方 TreeLearn 改变了哪些预测。

每个适配器 checkpoint 内的原始 TreeLearn 权重都已经 bitwise 验证不变，因此一次
forward 可以得到严格对齐的 base 与 adapted 输出。统计覆盖：

- tree→non-tree、non-tree→tree 翻转；
- 原始 base seed、适配后 seed、增加/移除的 seed；
- tree probability 改变量；
- XY/Z offset residual；
- HSCA/MLP gate；
- 按 8 个归一化高度层、verticality 和 0.6 m 三维体素占用密度分桶。

注意：统计基于 pipeline tiles，因此重叠区域会按其推理曝光次数计权。这不影响同一
domain 内 base/adapted 的严格配对，但不能把 points 数解释为森林唯一点数。
默认从每个 tile 确定性等距抽取 5000 点进行统计；模型仍对完整 tile forward，抽样只减少统计开销。

## 服务器 smoke test

```bash
cd ~/projects/zrx/code/TreeLearn
git pull --ff-only origin vertical-instance-quality
conda activate TreeLearn
mkdir -p logs/height_context_attention

python -m unittest \
  tests.test_height_context_diagnostics \
  tests.test_height_context_gate_output -v

python -u tools/diagnostics/diagnose_height_context_domain_drift.py \
  --max_scans 2 \
  --output_dir logs/height_context_attention/d1_domain_drift_smoke \
  2>&1 | tee logs/height_context_attention/d1_domain_drift_smoke.log
```

Smoke 输出必须包含 L1W/Wytham 与 A1/A2 四行，且所有数值有限。Smoke 结果不得
用于科学结论。

## 全量运行

```bash
nohup python -u \
  tools/diagnostics/diagnose_height_context_domain_drift.py \
  --output_dir logs/height_context_attention/d1_domain_drift \
  > logs/height_context_attention/d1_domain_drift_run.log \
  2>&1 < /dev/null &

echo $! | tee logs/height_context_attention/d1_domain_drift.pid
tail -f logs/height_context_attention/d1_domain_drift_run.log
```

完成后查看：

```bash
cat logs/height_context_attention/d1_domain_drift/summary.md
```

完整分桶数据位于：

```text
logs/height_context_attention/d1_domain_drift/summary.json
```

## 决策规则

- Wytham 的 tree→non-tree 或 seed removal 相对 L1W 明显放大：下一阶段采用
  semantic/seed identity preservation；
- Wytham offset residual 明显放大而 semantic flip 稳定：采用 offset residual
  正则与 teacher consistency；
- gate 与 residual 同时跨域升高：增加 gate calibration/identity regularization；
- A1/A2 漂移相近：问题主要来自训练目标，不是注意力结构；
- A2 特有漂移：关闭现有 attention mixer，不再扩大它。

D1 完成后才设计 D2。D2 只能使用固定 train/validation forests 上的 SODA 合成退化
选择 checkpoint，不得根据 Wytham GT 指标选择 loss 权重。
