# P0b 层次化 Superproposal Oracle

- 直接复用 P0 固定验证集 artifacts；未重新运行 B1，未读取 Wytham。
- 三种预聚合均不读取 GT；GT 只用于预聚合完成后的 Oracle 验收。
- P0 的失败结论保持不变，本阶段检验新的层次化实例形成假设。

| Method | Evaluated | Nodes | Compression | Inflation | Recall | Commission | F1 | F1 gain | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| xy_block | 2/5 | 2,594 | 119.05x | 2.80x | 59.468% | 50.779% | 53.862% | -38.609 pp | early-stop |
| vertical_profile | 2/5 | 173,819 | 1.86x | 187.71x | 63.123% | 61.796% | 47.599% | -44.872 pp | early-stop |
| adaptive_component | 0/5 | 27,537 | 0.00x | 13.94x | 0.000% | 0.000% | 0.000% | +0.000 pp | early-stop |

## 提前淘汰

- xy_block: 2 negative forests already make the fixed 4/5 plot-consistency Gate unreachable.
- vertical_profile: 2 negative forests already make the fixed 4/5 plot-consistency Gate unreachable.
- adaptive_component: adaptive_component/G4N still has 27,537 nodes, exceeding the global P0b maximum 19,760 before the remaining forests are counted.

## 每森林 F1 gain

| Method | G4N | G4W | L1N | O1N | O1W |
|---|---:|---:|---:|---:|---:|
| xy_block | -40.412 pp | -37.720 pp | - | - | - |
| vertical_profile | -51.099 pp | -41.547 pp | - | - | - |
| adaptive_component | - | - | - | - | - |

## 决策

- 推荐方法：**None**
- Primary Gate：**False**

STOP：层次化预聚合仍无足够上限，关闭 proposal-relation 路线。
