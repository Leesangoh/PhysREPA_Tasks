# OOD Pairwise Bootstrap

- task: `drawer`
- summary: `/home/solee/physrepa_tasks/artifacts/results/functional_significance_drawer_fusion_ood_l1_summary.json`
- resamples: `1000`
- bootstrap seed: `42`

| Comparison | Mean gap | 95% CI | Episode count |
| --- | ---: | ---: | ---: |
| `vjepa_fusion - vjepa_last` | 0.0061 | [0.0053, 0.0068] | 300 |
| `vjepa_fusion - vjepa_pez` | 0.0078 | [0.0068, 0.0089] | 300 |
| `vjepa_fusion - videomae_best` | -0.0090 | [-0.0099, -0.0079] | 300 |
| `videomae_best - dino_best` | 0.0367 | [0.0338, 0.0396] | 300 |
