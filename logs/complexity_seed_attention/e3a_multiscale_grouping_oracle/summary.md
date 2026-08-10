# E3a 复杂度感知多尺度分组 Proposal Oracle

- 数据：固定五个 validation forests；未读取 Wytham。
- 不修改 TreeLearn votes、种子集合或特征。
- 固定 HDBSCAN `min_cluster_size`：[25, 35, 50, 75, 100]
- 原版尺度：50
- 候选种子：1,908,145
- GT 树：1,757

## Macro 单尺度与 Proposal Oracle

| Mode | Tree recall | Commission | F1 |
|---|---:|---:|---:|
| mcs_25 | 96.257% | 37.766% | 73.598% |
| mcs_35 | 96.301% | 34.075% | 76.506% |
| mcs_50 (base) | 96.698% | 30.199% | 79.577% |
| mcs_75 | 96.673% | 27.278% | 81.774% |
| mcs_100 | 96.632% | 24.930% | 83.488% |
| proposal_oracle | 97.136% | 0.000% | 98.543% |

## Oracle 效果

- Tree recall 增益：+0.438 pp
- F1 上限增益：+18.967 pp
- 新恢复 GT 树：7
- 有新增恢复的森林：3/5
- 最佳尺度非原版的 GT 比例：18.213%
- 平均 proposal 膨胀倍数：5.086x

## 每森林

| Plot | Base recall | Oracle recall | Gain | Recovered | Non-base best | Inflation |
|---|---:|---:|---:|---:|---:|---:|
| G4N | 98.033% | 98.361% | +0.328 pp | 1 | 14.098% | 5.091x |
| G4W | 97.227% | 97.574% | +0.347 pp | 2 | 14.558% | 5.012x |
| L1N | 97.566% | 97.566% | +0.000 pp | 0 | 18.805% | 5.042x |
| O1N | 97.484% | 97.484% | +0.000 pp | 0 | 11.321% | 5.089x |
| O1W | 93.182% | 94.697% | +1.515 pp | 4 | 34.091% | 5.196x |

## Gate

- expected_validation_plots: **True**
- baseline_reproduced: **True**
- scale_grid_locked: **True**
- oracle_recall_gain_passed: **False**
- recovered_tree_count_passed: **False**
- per_plot_recovery_passed: **True**
- scale_diversity_passed: **True**
- proposal_inflation_passed: **True**
- passed: **False**

STOP：多尺度分组没有足够的验证集恢复上限；关闭复杂度感知分组路线，不使用 Wytham 调参。
