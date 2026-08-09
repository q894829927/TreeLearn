# E2 固定 50 实例 artifact 审计

- 样本数：50
- 地块数：14
- split：train, validation
- 一致性错误：0

## 类别分布

- ambiguous: 12
- negative: 16
- positive: 22

## Gate

- enough_samples: **True**
- enough_plots: **True**
- train_present: **True**
- validation_present: **True**
- positive_present: **True**
- negative_present: **True**
- ambiguous_present: **True**
- edge_present: **True**
- all_rows_audited: **True**
- all_consistency_checks_passed: **True**
- passed: **True**

该审计逐条验证 CSV/NPZ、标签阈值、边界有效性、来源 split、8 层占比和空层数值。它不重新生成逐点候选几何；IoU 算法的逐点正确性由 E1 Oracle 对齐检查和 E2 单元测试覆盖。
