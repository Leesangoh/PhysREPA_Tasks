# Phase 1 Results: push / giant

## Summary

| target | output_dim | L0 | L8 | peak_r2 | peak_layer | last | class |
|---|---:|---:|---:|---:|---:|---:|---|
| ee_direction_3d | 3 | 0.5211 | 0.7669 | 0.8153 | 24 | 0.7983 | PEZ-like |
| ee_speed | 1 | 0.5175 | 0.9134 | 0.9304 | 25 | 0.9179 | intermediate |

## Hypothesis checks

- H1 (controls high from L0): `FAIL`
- H2 (static physics PEZ-like): `FAIL`

## Sanity checks

- feature_root: `/mnt/md1/solee/features/physprobe_vitg_tokenpatch/push`
- feature_type: `token_patch`
- run_tag: `scale_giant_seed42`
- n_episodes: `1500`
- num_layers: `40`
- feature_dim_layer0: `360448`
- window_count_mode: `58`
- missing_feature_keys: `0`
- patch_shape: `[256, 1408]`
- groupkfold_overlap_zero: `True`
- gt_distribution_plot: `/home/solee/physrepa_tasks/artifacts/figures/gt_distribution_push.png`
- control_targets_l0_ge_0_8: `False`
- gt_summary: `{'ee_direction_3d': {'shape': [1500, 3], 'var': 0.031199700141461905, 'finite': True}, 'ee_speed': {'shape': [1500], 'var': 0.0017155153710907913, 'finite': True}}`

## Next-step decision

- Control targets did not meet the pre-registered expectation. Inspect pipeline before escalating.
