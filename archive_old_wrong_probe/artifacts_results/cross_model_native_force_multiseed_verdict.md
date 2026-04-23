# Cross-Model Native Force Multiseed Verdict

Matched evaluation on the recollected `Strike` dataset using native `physics_gt.contact_force` rather than the surrogate force proxy.

## Summary Table

| Model | Seeds | Peak $R^2$ | 95% bootstrap CI | Peak layer | Peak depth | 95% depth CI | $L0$ | $L8$ | Last |
| --- | ---: | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: |
| V-JEPA~2 Large | 3 | 0.566 ± 0.002 | [0.564, 0.567] | 16.7 ± 0.6 | 0.694 ± 0.024 | [0.667, 0.708] | 0.262 | 0.457 | 0.529 |
| VideoMAE-L | 3 | 0.531 ± 0.001 | [0.530, 0.532] | 22.0 ± 1.0 | 0.917 ± 0.042 | [0.875, 0.958] | 0.189 | 0.445 | 0.528 |
| DINOv2-L | 3 | 0.205 ± 0.005 | [0.200, 0.210] | 17.3 ± 0.6 | 0.722 ± 0.024 | [0.708, 0.750] | -0.063 | 0.030 | 0.161 |

## Paired Bootstrap Comparisons

| Comparison | Mean peak-$R^2$ delta | 95% bootstrap CI | Tail $p$ | CI excludes 0? |
| --- | ---: | --- | ---: | --- |
| V-JEPA~2 Large - VideoMAE-L | 0.034 | [0.033, 0.036] | 0.0000 | yes |
| V-JEPA~2 Large - DINOv2-L | 0.360 | [0.355, 0.367] | 0.0000 | yes |

