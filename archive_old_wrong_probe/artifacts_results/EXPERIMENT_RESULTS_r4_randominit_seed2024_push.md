# Phase 1 Results: push / large

## Summary

| target | output_dim | L0 | L8 | peak_r2 | peak_layer | last | class |
|---|---:|---:|---:|---:|---:|---:|---|
| ee_direction_3d | 3 | 0.5482 | 0.5784 | 0.5842 | 16 | 0.5823 | intermediate |
| ee_speed | 1 | 0.5682 | 0.6157 | 0.6291 | 13 | 0.6210 | PEZ-like |

## Hypothesis checks

- H1 (controls high from L0): `FAIL`
- H2 (static physics PEZ-like): `FAIL`

## Sanity checks

- feature_root: `/mnt/md1/solee/features/physprobe_vitl_tokenpatch_randominit_seed0/push`
- feature_type: `token_patch`
- run_tag: `r4_randominit_seed2024`
- n_episodes: `1500`
- num_layers: `24`
- feature_dim_layer0: `262144`
- window_count_mode: `58`
- missing_feature_keys: `0`
- patch_shape: `[256, 1024]`
- groupkfold_overlap_zero: `True`
- gt_distribution_plot: `/home/solee/physrepa_tasks/artifacts/figures/gt_distribution_push.png`
- control_targets_l0_ge_0_8: `False`
- gt_summary: `{'ee_direction_3d': {'shape': [1500, 3], 'var': 0.031199700141461905, 'finite': True}, 'ee_speed': {'shape': [1500], 'var': 0.0017155153710907913, 'finite': True}}`

## Next-step decision

- Control targets did not meet the pre-registered expectation. Inspect pipeline before escalating.
