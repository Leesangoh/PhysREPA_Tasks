# Phase 1 Results: push / dinov2_large

## Summary

| target | output_dim | L0 | L8 | peak_r2 | peak_layer | last | class |
|---|---:|---:|---:|---:|---:|---:|---|
| ee_direction_3d | 3 | -1.4554 | 0.7005 | 0.7762 | 15 | 0.7651 | intermediate |
| ee_speed | 1 | -1.1356 | 0.7294 | 0.8854 | 23 | 0.8854 | intermediate |

## Hypothesis checks

- H1 (controls high from L0): `FAIL`
- H2 (static physics PEZ-like): `FAIL`

## Sanity checks

- feature_root: `/mnt/md1/solee/features/physprobe_dinov2_large_tokenpatch/push`
- feature_type: `token_patch`
- run_tag: `cross_dinov2_large_seed42`
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
