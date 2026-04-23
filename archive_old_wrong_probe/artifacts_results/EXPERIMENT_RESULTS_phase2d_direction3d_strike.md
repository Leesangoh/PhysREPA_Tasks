# Phase 1 Results: strike / large

## Summary

| target | output_dim | L0 | L8 | peak_r2 | peak_layer | last | class |
|---|---:|---:|---:|---:|---:|---:|---|
| ee_direction_3d | 3 | 0.6226 | 0.8180 | 0.8491 | 22 | 0.8475 | intermediate |
| object_direction_3d | 3 | 0.5209 | 0.7738 | 0.8132 | 12 | 0.8125 | PEZ-like |

## Hypothesis checks

- H1 (controls high from L0): `FAIL`
- H2 (static physics PEZ-like): `FAIL`

## Sanity checks

- feature_root: `/mnt/md1/solee/features/physprobe_vitl_tokenpatch/strike`
- feature_type: `token_patch`
- run_tag: `phase2d_direction3d`
- n_episodes: `2895`
- num_layers: `24`
- feature_dim_layer0: `262144`
- window_count_mode: `58`
- missing_feature_keys: `0`
- patch_shape: `[256, 1024]`
- groupkfold_overlap_zero: `True`
- gt_distribution_plot: `/home/solee/physrepa_tasks/artifacts/figures/gt_distribution_strike.png`
- control_targets_l0_ge_0_8: `False`
- gt_summary: `{'ee_direction_3d': {'shape': [2895, 3], 'var': 0.013947314553759792, 'finite': True}, 'object_direction_3d': {'shape': [2895, 3], 'var': 0.13210323903609766, 'finite': True}}`

## Next-step decision

- Control targets did not meet the pre-registered expectation. Inspect pipeline before escalating.
