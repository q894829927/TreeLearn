# Predicted-instance TP/FP separability

## Counts

| Status | Count |
|---|---:|
| unmatched_ignored | 1163 |
| tp | 568 |
| fp_counted | 131 |

## Best features

| Feature | Separability AUC | Direction | TP median | FP median |
|---|---:|---|---:|---:|
| point_density_bbox | 0.861036 | higher for TP | 246.802614 | 53.115012 |
| confidence_p90 | 0.852536 | higher for TP | 0.855000 | 0.805000 |
| base_vote_radius_rms | 0.840152 | lower for TP | 0.291978 | 0.687980 |
| confidence_mean | 0.820839 | higher for TP | 0.743382 | 0.618840 |
| num_points | 0.819011 | higher for TP | 13116.500000 | 2398.000000 |
| confidence_p10 | 0.811237 | higher for TP | 0.595000 | 0.295000 |
| offset_z_abs_mean | 0.810692 | higher for TP | 12.895398 | 6.520452 |
| seed_confidence_mean | 0.798772 | higher for TP | 0.803749 | 0.741933 |
| confidence_p50 | 0.796836 | higher for TP | 0.785000 | 0.705000 |
| semantic_probability_p10 | 0.794988 | higher for TP | 0.995000 | 0.745000 |
| seed_fraction | 0.794498 | lower for TP | 0.027976 | 0.078933 |
| seed_confidence_p10 | 0.792919 | higher for TP | 0.785000 | 0.535000 |
| semantic_probability_mean | 0.791071 | higher for TP | 0.996920 | 0.926723 |
| seed_confidence_p50 | 0.775239 | higher for TP | 0.805000 | 0.795000 |
| confidence_std | 0.771826 | lower for TP | 0.118733 | 0.179867 |

## Gate

- enough_false_positives: **True**
- confidence_auc_passed: **True**
- passed: **True**
