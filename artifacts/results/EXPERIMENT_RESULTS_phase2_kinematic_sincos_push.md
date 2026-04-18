# Phase 1 Results: push / large

## Summary

| target | output_dim | L0 | L8 | peak_r2 | peak_layer | last | class |
|---|---:|---:|---:|---:|---:|---:|---|
| ee_direction_sincos | 2 | 0.5554 | 0.7894 | 0.8068 | 13 | 0.7969 | PEZ-like |
| object_direction_sincos | 2 | -0.2202 | 0.0483 | 0.0841 | 23 | 0.0841 | never-linear |

## Hypothesis checks

- H1 (controls high from L0): `FAIL`
- H2 (static physics PEZ-like): `FAIL`

## Sanity checks

- feature_root: `/mnt/md1/solee/features/physprobe_vitl_tokenpatch/push`
- feature_type: `token_patch`
- run_tag: `phase2_kinematic_sincos`
- n_episodes: `1500`
- num_layers: `24`
- feature_dim_layer0: `262144`
- window_count_mode: `58`
- missing_feature_keys: `0`
- patch_shape: `[256, 1024]`
- groupkfold_overlap_zero: `True`
- gt_distribution_plot: `/home/solee/physrepa_tasks/artifacts/figures/gt_distribution_push.png`
- control_targets_l0_ge_0_8: `False`
- gt_summary: `{'ee_direction_sincos': {'shape': [1500, 2], 'var': 0.06027721993100113, 'finite': True}, 'object_direction_sincos': {'shape': [1500, 2], 'var': 0.20749827003389973, 'finite': True}}`

## Next-step decision

- Control targets did not meet the pre-registered expectation. Inspect pipeline before escalating.
