# T0 Sparse U-Net enhancement smoke

| Model | Identity error | Params added | Grad tensors | Peak GB | Forward s | Pass |
|---|---:|---:|---:|---:|---:|---|
| identity | 0.000e+00 | 0 | 0/0 | 0.023 | 0.0040 | True |
| residual_adapter | 0.000e+00 | 1,143 | 19/19 | 0.024 | 0.0045 | True |
| sparse_se | 0.000e+00 | 971 | 15/15 | 0.026 | 0.0047 | True |
| selective_kernel | 0.000e+00 | 46,278 | 14/14 | 0.026 | 0.0053 | True |
| hcag | 0.000e+00 | 646 | 22/22 | 0.026 | 0.0050 | True |
| window_attention | 0.000e+00 | 4,969 | 15/15 | 0.034 | 0.0061 | True |

## Gate

- all_candidates_passed: **True**
- memory_limit_gb: **23.0**
- passed: **True**
