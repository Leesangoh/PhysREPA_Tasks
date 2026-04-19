# Phase 1 Results: push / large

## Summary

| target | output_dim | L0 | L8 | peak_r2 | peak_layer | last | class |
|---|---:|---:|---:|---:|---:|---:|---|
| ee_direction_3d | 3 | 0.6522 | 0.8056 | 0.8172 | 11 | 0.8128 | PEZ-like |
| object_direction_3d | 3 | -0.1602 | 0.0968 | 0.1355 | 23 | 0.1355 | never-linear |

## Hypothesis checks

- H1 (controls high from L0): `FAIL`
- H2 (static physics PEZ-like): `FAIL`

## Sanity checks

- feature_root: `/mnt/md1/solee/features/physprobe_vitl_tokenpatch/push`
- feature_type: `token_patch`
- run_tag: `phase2d_direction3d`
- n_episodes: `1500`
- num_layers: `24`
- feature_dim_layer0: `262144`
- window_count_mode: `58`
- missing_feature_keys: `0`
- patch_shape: `[256, 1024]`
- groupkfold_overlap_zero: `True`
- gt_distribution_plot: `/home/solee/physrepa_tasks/artifacts/figures/gt_distribution_push.png`
- control_targets_l0_ge_0_8: `False`
- gt_summary: `{'ee_direction_3d': {'shape': [1500, 3], 'var': 0.031199700141461905, 'finite': True}, 'object_direction_3d': {'shape': [1500, 3], 'var': 0.16791827166705872, 'finite': True}}`

## Next-step decision

- Control targets did not meet the pre-registered expectation. Inspect pipeline before escalating.
