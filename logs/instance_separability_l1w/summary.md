# Predicted-instance TP/FP separability

## Counts

| Status | Count |
|---|---:|
| unmatched_ignored | 210 |
| tp | 156 |
| fp_counted | 5 |

## Best features

| Feature | Separability AUC | Direction | TP median | FP median |
|---|---:|---|---:|---:|
| seed_fraction | 1.000000 | lower for TP | 0.019366 | 0.174349 |
| point_density_bbox | 1.000000 | higher for TP | 422.271684 | 22.817814 |
| num_points | 0.998718 | higher for TP | 55250.000000 | 499.000000 |
| semantic_probability_std | 0.989744 | lower for TP | 0.010719 | 0.126699 |
| confidence_p90 | 0.989103 | higher for TP | 0.875000 | 0.795000 |
| semantic_probability_mean | 0.975641 | higher for TP | 0.999467 | 0.902956 |
| num_clustered_seeds | 0.971795 | higher for TP | 1127.000000 | 87.000000 |
| bbox_area | 0.964103 | higher for TP | 129.840363 | 40.820408 |
| confidence_mean | 0.946154 | higher for TP | 0.741422 | 0.609152 |
| height | 0.944872 | higher for TP | 33.240000 | 21.860000 |
| offset_z_abs_mean | 0.938462 | higher for TP | 21.916278 | 4.340481 |
| seed_confidence_mean | 0.924359 | higher for TP | 0.811367 | 0.783425 |
| seed_confidence_p90 | 0.920513 | higher for TP | 0.825000 | 0.795000 |
| y_span | 0.919231 | higher for TP | 11.400000 | 6.350000 |
| confidence_p10 | 0.901923 | higher for TP | 0.575000 | 0.385000 |

## Gate

- enough_false_positives: **False**
- confidence_auc_passed: **True**
- passed: **False**
