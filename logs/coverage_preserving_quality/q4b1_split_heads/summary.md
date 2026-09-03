# Q4b1 Candidate + K + Safety 三头 MLP

- 数据：固定 13 个 train forests 与 5 个 validation forests。
- Wytham 未参与训练、checkpoint 或阈值选择。
- 参数量：18,821

| Seed | Epoch | Candidate AP | K accuracy | Safety AP | Accepted | F1 gain | Completeness drop | Commission increase | Pass |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 30 | 0.112044 | 0.898551 | 0.320324 | 16 | +0.061 pp | -0.337 pp | +0.178 pp | False |
| 43 | 25 | 0.095744 | 0.913043 | 0.296168 | 20 | +0.097 pp | -0.449 pp | +0.207 pp | False |
| 44 | 10 | 0.087485 | 0.898551 | 0.195895 | 21 | +0.074 pp | -0.449 pp | +0.252 pp | False |

## Aggregate

- F1 gain mean：+0.077 pp
- F1 gain std：0.018 pp

## Gate

- data_gate_passed: **True**
- locked_seed_present: **True**
- locked_seed_passed: **False**
- enough_seed_passes: **False**
- f1_seed_stability_passed: **True**
- passed: **False**

STOP：三头 MLP 不可部署，关闭实例拆分路线。
