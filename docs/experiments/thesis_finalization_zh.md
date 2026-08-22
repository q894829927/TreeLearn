# TreeLearn 毕业论文实验最终状态与收口方案

更新日期：2026-08-22

## 1. 冻结原则

从本日期起冻结模型、checkpoint、Wytham 设置、HDBSCAN、seed 策略和所有
后处理阈值。Wytham 已被用于锁定后的最终探索性评价，禁止根据其结果继续
调整模型或选择性汇报。

关闭的路线包括：

- seed attention / vote refinement；
- learned merge、fragment merge 和 instance split；
- seed completion；
- Sparse U-Net 多模型竞赛的进一步训练；
- proposal-relation attention 及其层次化 superproposal 变体。

## 2. Sparse U-Net 最终结果

### 固定五森林 validation（seed 42）

| Model | Macro F1 | Completeness | Commission | 相对 B1 F1 |
|---|---:|---:|---:|---:|
| B0 Official | 91.640% | 93.702% | 10.165% | -0.605 pp |
| B1 Partial Fine-tune | 92.245% | 92.152% | 7.600% | 0 |
| M4 Window Attention | 92.311% | 91.834% | 7.171% | +0.066 pp |

M4 未达到预注册的 +0.2 pp T1 Gate，只有 3/5 森林非负，因此不能进入
“注意力有效”的确认性结论。

### Wytham 锁定探索性评价

| Model | Completeness | Commission | Detection F1 | Precision | Recall | Coverage |
|---|---:|---:|---:|---:|---:|---:|
| B0 Official | 64.8% | 18.9% | 72.0% | 62.5% | 80.5% | 57.7% |
| B1 Partial Fine-tune | 59.7% | 13.1% | 70.8% | 56.4% | 71.7% | 53.0% |
| M4 Window Attention | 59.7% | 13.0% | 70.9% | 56.1% | 71.4% | 52.7% |

M4 相对 B1 仅约 +0.1 pp Detection F1，并且 Precision、Recall、Coverage
均略低；相对官方 B0 则约 -1.1 pp F1、-5.1 pp Completeness 和 -5.0 pp
Coverage。B1/M4 都表现为降低 Commission、同时损失更多 Completeness，
不能称为复杂森林精度提升。

## 3. 已成立的主要正结果

当前最可靠的正结果是冻结 Vertical-MLP 实例质量评分提供的选择性风险控制：

- L1W：Commission-area 相对随机排序降低 77.855%，F1-area 增加
  10.962 pp；
- Wytham：Commission-area 相对随机排序降低 41.322%，F1-area 增加
  10.293 pp；
- Wytham 随机排序置换检验中两个指标均为 p=0.0001；
- 锁定 keep ratio 0.85 时，Wytham F1 从 72.081% 提升到 73.232%，
  Commission 从 18.741% 降到 12.893%，代价是 Completeness 下降
  1.596 pp。

必须同时报告完整风险—覆盖曲线和 Completeness 代价，不能只报告最佳 F1。

## 4. 论文可用结论

允许：

1. 参数受控的 Sparse U-Net 注意力竞赛显示，局部窗口注意力是四种候选中
   最稳定者，并轻微改善 validation F1/Commission；
2. 该增益未通过预设 Gate，且没有转化为 Wytham 总体精度提升；
3. 实例质量排序在源域与复杂跨域森林中均显著优于随机选择，能提供可调的
   Commission—Completeness 风险控制；
4. 多组 Oracle 和可部署性实验揭示 TreeLearn 错误主要受实例形成表示与
   跨域 seed/semantic 偏移限制。

禁止：

- “Window Attention 显著优于 TreeLearn”；
- “注意力使 Wytham 单木分割精度提高”；
- 从 Wytham 结果重新选择 checkpoint、结构或阈值；
- 隐藏失败的 B1、M4、P0/P0b 或 E10 结果。

## 5. 建议论文结构

1. 绪论与相关工作；
2. TreeLearn 基线、数据集和统一评价协议；
3. Sparse U-Net 注意力模块与受控结构竞赛；
4. 垂直结构实例质量估计与选择性风险控制（主要方法章）；
5. 误差 Oracle、可部署性边界与复杂森林跨域实验；
6. 讨论、局限性与结论。

如果题目允许调整，建议使用：

> 基于稀疏深度网络的森林点云单木分割与实例质量控制研究

如果题目必须保留“注意力机制”，建议使用：

> 融合局部注意力与实例质量控制的森林点云单木分割研究

其中“局部注意力”必须作为结构研究和消融，不作为已证明的性能主贡献。

## 6. 下一阶段

不再训练模型。下一阶段只做：

1. 自动汇总最终结果表；
2. 生成 validation 与 Wytham 对比图；
3. 整理所有 Gate 和失败路线流程图；
4. 开始撰写方法、实验和讨论章节；
5. 对论文中的每个性能主张建立对应日志/summary 证据索引。
