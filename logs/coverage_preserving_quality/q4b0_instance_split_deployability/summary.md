# Q4b0 Raw-XY 实例拆分可部署性分解

- 数据：固定五个 validation forests；未读取 Wytham。
- 候选父实例仍由 Q3 GT 类别给出，本阶段不代表完整可部署系统。
- 目的：分别移除 GT 验收与 GT 子树数量 K。
- 欠分割 GT：73
- 候选父实例：69

## 汇总指标

| Mode | Proposed | Accepted | Recovered | Lost | Completeness | Commission | F1 | F1 gain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | - | - | - | - | 91.639% | 14.413% | 88.509% | - |
| known_k_oracle_accept | 69 | 25 | 22 | 0 | 93.042% | 14.138% | 89.308% | +0.798 pp |
| known_k_accept_all | 69 | 69 | 22 | 6 | 92.705% | 16.142% | 88.060% | -0.450 pp |
| fixed_k2_accept_all | 69 | 69 | 22 | 6 | 92.705% | 16.100% | 88.083% | -0.426 pp |

## 每森林 F1 差值

| Plot | Known-K accept-all | Fixed-K2 accept-all |
|---|---:|---:|
| G4N | +0.379 pp | +0.379 pp |
| G4W | -0.734 pp | -0.657 pp |
| L1N | -0.403 pp | -0.403 pp |
| O1N | -1.435 pp | -1.435 pp |
| O1W | -0.111 pp | -0.111 pp |

## 完整性 Gate

- expected_validation_plots: **True**
- q4a_main_gate_passed: **True**
- q4a_recommended_simple_geometry: **True**
- baseline_reproduced: **True**
- known_k_oracle_reproduced: **True**
- passed: **True**

## known_k_accept_all Gate

- f1_gain_passed: **False**
- completeness_preserved: **True**
- commission_preserved: **False**
- plot_consistency_passed: **False**
- passed: **False**

## fixed_k2_accept_all Gate

- f1_gain_passed: **False**
- completeness_preserved: **True**
- commission_preserved: **False**
- plot_consistency_passed: **False**
- passed: **False**

## 决策

- 推荐路线：**candidate_child_count_and_safety_heads**
- 全部接受未通过；Q4b1 必须同时训练候选、K 与安全验收头。

下一阶段仍只使用固定 train/validation forests，不得读取 Wytham。
