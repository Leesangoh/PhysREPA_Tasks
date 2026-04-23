# OOD Pairwise Bootstrap

- task: `push`
- summary: `/home/solee/physrepa_tasks/artifacts/results/functional_significance_push_fusion_ood_l1_summary.json`
- resamples: `1000`
- bootstrap seed: `42`

| Comparison | Mean gap | 95% CI | Episode count |
| --- | ---: | ---: | ---: |
| `vjepa_fusion - vjepa_last` | -0.0005 | [-0.0040, 0.0035] | 911 |
| `vjepa_fusion - vjepa_pez` | 0.0241 | [0.0194, 0.0296] | 911 |
| `vjepa_fusion - videomae_best` | 0.0064 | [-0.0014, 0.0161] | 911 |
| `videomae_best - dino_mid` | 0.1518 | [0.1320, 0.1712] | 911 |
