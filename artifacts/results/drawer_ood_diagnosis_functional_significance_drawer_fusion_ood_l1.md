# Drawer OOD Diagnosis

- summary: `/home/solee/physrepa_tasks/artifacts/results/functional_significance_drawer_fusion_ood_l1_summary.json`
- split: `/home/solee/physrepa_tasks/artifacts/results/functional_significance_drawer_fusion_ood_l1_split.json`
- chunk_len: `8`

## Per-episode action-regression R²

- `vjepa_pez`: IID mean 0.8704, OOD mean 0.8874, median shift 0.0086
- `vjepa_last`: IID mean 0.8720, OOD mean 0.8891, median shift 0.0054
- `vjepa_fusion`: IID mean 0.8773, OOD mean 0.8952, median shift 0.0083
- `videomae_best`: IID mean 0.8885, OOD mean 0.9042, median shift 0.0093
- `dino_best`: IID mean 0.8496, OOD mean 0.8675, median shift 0.0114

## Action statistics

| Metric | IID mean | OOD mean |
| --- | ---: | ---: |
| `episode_len` | 295.0000 | 295.0000 |
| `action_variance` | 0.8407 | 0.8555 |
| `action_autocorr_lag1` | 0.9013 | 0.9060 |
| `persistence_chunk_r2` | 0.2255 | 0.2591 |
