# Surrogate vs Native Force Comparison

Matched comparison of the earlier surrogate `contact_force_proxy` panel and the recollected native `contact_force` panel on Strike.

| Model | Surrogate peak $R^2$ | Surrogate depth | Native peak $R^2$ | Native depth |
| --- | ---: | ---: | ---: | ---: |
| V-JEPA~2 Large | 0.211 ± 0.005 | 0.847 ± 0.024 | 0.566 ± 0.002 | 0.694 ± 0.024 |
| VideoMAE-L | 0.206 ± 0.012 | 0.806 ± 0.024 | 0.531 ± 0.001 | 0.917 ± 0.042 |
| DINOv2-L | 0.153 ± 0.002 | 0.819 ± 0.120 | 0.205 ± 0.005 | 0.722 ± 0.024 |

## Bootstrap Comparisons

| Comparison | Surrogate delta | Surrogate 95% CI | Native delta | Native 95% CI |
| --- | ---: | --- | ---: | --- |
| V-JEPA~2 Large - VideoMAE-L | 0.005 | [-0.015, 0.014] | 0.034 | [0.033, 0.036] |
| V-JEPA~2 Large - DINOv2-L | 0.058 | [0.055, 0.060] | 0.360 | [0.355, 0.367] |

