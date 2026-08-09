# E5 Vertical-Attention

- Instances: 9402
- Vertical tokens: 8 x 72
- Trainable parameters: 52,835
- Paired seed wins: 0/3

| Model | ROC-AUC | FP AP | IoU MAE | Completeness | Commission | Filtered F1 |
|---|---:|---:|---:|---:|---:|---:|
| vertical_mlp | 0.995627 | 0.991869 | 0.086063 | 0.990494 | 0.037738 | 0.976172 |
| vertical_attention | 0.995634 | 0.991556 | 0.086244 | 0.996900 | 0.242638 | 0.853377 |

## Delta: Attention minus Vertical-MLP

- Filtered F1: -0.122795
- Commission reduction: -0.204900
- Completeness drop: -0.006406
- FP AP: -0.000313
- IoU MAE reduction: -0.000182
- Validation inference: 0.000413 ms/instance

## Paired seeds

- seed 42: 0.977071 -> 0.793019, won=False
- seed 43: 0.974687 -> 0.793019, won=False
- seed 44: 0.976758 -> 0.974093, won=False

## Gate

- e4_baseline_passed: **True**
- mean_f1_gain_passed: **False**
- commission_tradeoff_passed: **False**
- fp_ap_not_decreased: **False**
- paired_seed_wins_passed: **False**
- effect_size_passed: **False**
- runtime_gate_deferred_to_e6: **True**
- passed: **False**
