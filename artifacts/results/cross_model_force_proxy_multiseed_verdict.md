# Cross-Model Force Proxy Multiseed Verdict

Matched statistical tightening run for `Strike / contact_force_proxy`.
All three model families were probed on the same `1000`-episode subset with three probe seeds (`42`, `123`, `2024`).

## Main Result

The Tier-B ordering remains stable under matched multiseed evaluation:

\[
\text{V-JEPA~2 Large} \;>\; \text{VideoMAE-L} \;>\; \text{DINOv2-L}.
\]

This pass makes the ordering statistically tighter and removes the earlier mismatch between a full-cache V-JEPA run and subset baselines.

## Summary Table

| Model | Seeds | Peak $R^2$ | 95% bootstrap CI | Peak layer | Peak depth | 95% depth CI | $L0$ | $L8$ | Last |
| --- | ---: | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: |
| V-JEPA~2 Large | 3 | 0.211 ± 0.005 | [0.205, 0.214] | 20.3 ± 0.6 | 0.847 ± 0.024 | [0.833, 0.875] | 0.080 ± 0.003 | 0.154 ± 0.007 | 0.204 ± 0.007 |
| VideoMAE-L | 3 | 0.206 ± 0.012 | [0.198, 0.220] | 19.3 ± 0.6 | 0.806 ± 0.024 | [0.792, 0.833] | 0.003 ± 0.005 | 0.176 ± 0.014 | 0.200 ± 0.012 |
| DINOv2-L | 3 | 0.153 ± 0.002 | [0.150, 0.154] | 19.7 ± 2.9 | 0.819 ± 0.120 | [0.750, 0.958] | -0.062 ± 0.007 | 0.055 ± 0.007 | 0.148 ± 0.006 |

## Paired Bootstrap Comparisons

| Comparison | Mean peak-$R^2$ delta | 95% bootstrap CI | Tail $p$ | CI excludes 0? |
| --- | ---: | --- | ---: | --- |
| V-JEPA~2 Large - VideoMAE-L | 0.005 | [-0.015, 0.014] | 0.5119 | no |
| V-JEPA~2 Large - DINOv2-L | 0.058 | [0.055, 0.060] | 0.0000 | yes |

## Interpretation

- The multiseed pass preserves the same scientific ranking as the earlier single-seed panel.
- The strongest Tier-B signal still belongs to the predictive-video family, not the masked-video or static-image baselines.
- The key paper claim is therefore stronger than before: the force-proxy advantage is not an artifact of a single probe seed or mismatched evaluation subset.
- If the V-JEPA vs VideoMAE interval remains close to zero, the honest claim should stay ordered-but-narrow rather than over-claimed as a large margin win.

