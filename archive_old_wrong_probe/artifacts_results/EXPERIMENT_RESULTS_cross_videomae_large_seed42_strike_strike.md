# Phase 1 Results: strike / videomae_large

## Summary

| target | output_dim | L0 | L8 | peak_r2 | peak_layer | last | class |
|---|---:|---:|---:|---:|---:|---:|---|
| object_direction_3d | 3 | 0.4498 | 0.7255 | 0.7877 | 23 | 0.7877 | intermediate |

## Hypothesis checks

- H1 (controls high from L0): `FAIL`
- H2 (static physics PEZ-like): `FAIL`

## Sanity checks

- feature_root: `/mnt/md1/solee/features/physprobe_videomae_large_tokenpatch/strike`
- feature_type: `token_patch`
- run_tag: `cross_videomae_large_seed42_strike`
- n_episodes: `3000`
- num_layers: `24`
- feature_dim_layer0: `200704`
- window_count_mode: `58`
- missing_feature_keys: `0`
- patch_shape: `[196, 1024]`
- groupkfold_overlap_zero: `True`
- gt_distribution_plot: `/home/solee/physrepa_tasks/artifacts/figures/gt_distribution_strike.png`
- control_targets_l0_ge_0_8: `False`
- gt_summary: `{'object_direction_3d': {'shape': [3000, 3], 'var': 0.1323175571157177, 'finite': True}}`

## Next-step decision

- Control targets did not meet the pre-registered expectation. Inspect pipeline before escalating.
