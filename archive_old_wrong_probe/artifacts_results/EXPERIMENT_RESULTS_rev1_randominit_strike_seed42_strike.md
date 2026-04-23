# Phase 1 Results: strike / large

## Summary

| target | output_dim | L0 | L8 | peak_r2 | peak_layer | last | class |
|---|---:|---:|---:|---:|---:|---:|---|
| ee_direction_3d | 3 | 0.6069 | 0.6132 | 0.6215 | 23 | 0.6215 | intermediate |
| object_direction_3d | 3 | 0.4961 | 0.4928 | 0.5142 | 2 | 0.5079 | intermediate |
| ee_speed | 1 | 0.7708 | 0.7949 | 0.8170 | 4 | 0.8016 | intermediate |

## Hypothesis checks

- H1 (controls high from L0): `FAIL`
- H2 (static physics PEZ-like): `FAIL`

## Sanity checks

- feature_root: `/mnt/md1/solee/features/physprobe_vitl_tokenpatch_randominit_seed0/strike`
- feature_type: `token_patch`
- run_tag: `rev1_randominit_strike_seed42`
- n_episodes: `3000`
- num_layers: `24`
- feature_dim_layer0: `262144`
- window_count_mode: `58`
- missing_feature_keys: `0`
- patch_shape: `[256, 1024]`
- groupkfold_overlap_zero: `True`
- gt_distribution_plot: `/home/solee/physrepa_tasks/artifacts/figures/gt_distribution_strike.png`
- control_targets_l0_ge_0_8: `False`
- gt_summary: `{'ee_direction_3d': {'shape': [3000, 3], 'var': 0.013837192391244518, 'finite': True}, 'object_direction_3d': {'shape': [3000, 3], 'var': 0.1323175571157177, 'finite': True}, 'ee_speed': {'shape': [3000], 'var': 0.007819691353995543, 'finite': True}}`

## Next-step decision

- Control targets did not meet the pre-registered expectation. Inspect pipeline before escalating.
