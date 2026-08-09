# E3 global instance-quality baselines

- Instances: 9402
- Global features: 35
- Best aggregate model: global_mlp

| Model | ROC-AUC | FP AP | IoU MAE | Spearman | Filtered F1 |
|---|---:|---:|---:|---:|---:|
| single_feature | 0.972729 | 0.952152 | 0.164475 | 0.734634 | 0.793019 |
| logistic_regression | 0.992442 | 0.987715 | 0.118456 | 0.772575 | 0.793019 |
| global_mlp | 0.992655 | 0.987779 | 0.092260 | 0.815461 | 0.793019 |

## Gate

- aggregate_auc_passed: **True**
- aggregate_fp_ap_passed: **True**
- filtered_f1_not_below_unfiltered: **True**
- mlp_seed_stability_passed: **True**
- passed: **True**
