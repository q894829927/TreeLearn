# Q5b0 GT-free Seed-Completion Proposal Oracle

- 数据：固定五个 validation forests；未读取 Wytham。
- 候选区域完全由预测 vote、semantic、verticality 和 offset 生成。
- GT 只在评估端激活候选，用于测量候选生成上限。
- GT-free proposals：128,119
- Oracle 激活：41
- Proposal 覆盖目标树：31/31

| Mode | Completeness | Commission | F1 | F1 gain | Recovered | Lost |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 91.582% | 14.465% | 88.455% | - | - | - |
| proposal_oracle | 92.480% | 14.567% | 88.817% | +0.362 pp | 12 | 0 |

## 每森林

| Plot | Proposals | Covered | Recovered | Lost | F1 gain |
|---|---:|---:|---:|---:|---:|
| G4N | 23,819 | 13 | 3 | 0 | +0.622 pp |
| G4W | 34,206 | 11 | 9 | 0 | +0.658 pp |
| L1N | 26,390 | 2 | 0 | 0 | +0.000 pp |
| O1N | 18,615 | 1 | 0 | 0 | +0.000 pp |
| O1W | 25,089 | 4 | 0 | 0 | +0.213 pp |

## Q3 baseline reference audit

- Q5b0 effect sizes use the paired baseline from the same process.
- The older Q3 artifact is only a cross-run reference; accepted HDBSCAN count drift is shown explicitly.

| Plot | Expected TP/FP/FN | Observed TP/FP/FN | Drift | Exact |
|---|---:|---:|---:|---|
| G4N | 282/35/36 | 282/35/36 | +0/+0/+0 | True |
| G4W | 545/50/40 | 545/50/40 | +0/+0/+0 | True |
| L1N | 431/90/22 | 431/90/22 | +0/+0/+0 | True |
| O1N | 152/11/8 | 152/11/8 | +0/+0/+0 | True |
| O1W | 223/89/43 | 222/90/44 | -1/+1/+1 | False |

## Gate

- expected_validation_plots: **True**
- baseline_reference_per_plot_compatible: **True**
- baseline_reference_total_tp_drift_passed: **True**
- baseline_reference_total_fp_drift_passed: **True**
- baseline_reference_total_fn_drift_passed: **True**
- target_count_reproduced: **True**
- proposal_coverage_passed: **True**
- recovered_tree_count_passed: **True**
- f1_gain_passed: **True**
- lost_tree_count_passed: **True**
- commission_preserved: **True**
- plot_consistency_passed: **True**
- passed: **True**

PASS：进入 Q5b1，生成 train/validation proposal 数据并训练激活头。
