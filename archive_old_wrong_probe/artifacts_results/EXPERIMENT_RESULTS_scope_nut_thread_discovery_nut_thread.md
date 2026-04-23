# Phase 1 Results: nut_thread / large

## Summary

| target | output_dim | L0 | L8 | peak_r2 | peak_layer | last | class |
|---|---:|---:|---:|---:|---:|---:|---|
| ee_direction_3d | 3 | 0.0024 | 0.0684 | 0.1622 | 22 | 0.1526 | never-linear |
| ee_speed | 1 | 0.2067 | 0.4541 | 0.5390 | 21 | 0.4984 | intermediate |
| axial_progress | 1 | 0.2100 | 0.4773 | 0.6706 | 21 | 0.6380 | intermediate |
| nut_bolt_relative_angle | 1 | 0.1373 | 0.2917 | 0.4602 | 22 | 0.4548 | intermediate |

## Hypothesis checks

- H1 (controls high from L0): `FAIL`
- H2 (static physics PEZ-like): `FAIL`

## Sanity checks

- feature_root: `/mnt/md1/solee/features/physprobe_vitl/nut_thread`
- feature_type: `mean`
- run_tag: `scope_nut_thread_discovery`
- n_episodes: `2500`
- num_layers: `24`
- feature_dim_layer0: `1024`
- window_count_mode: `33`
- missing_feature_keys: `0`
- patch_shape: `None`
- groupkfold_overlap_zero: `True`
- gt_distribution_plot: `/home/solee/physrepa_tasks/artifacts/figures/gt_distribution_nut_thread.png`
- control_targets_l0_ge_0_8: `False`
- gt_summary: `{'ee_direction_3d': {'shape': [2500, 3], 'var': 0.005771191058736219, 'finite': True}, 'ee_speed': {'shape': [2500], 'var': 8.894412380707702e-07, 'finite': True}, 'axial_progress': {'shape': [2500], 'var': 1.3483129626257642e-07, 'finite': True}, 'nut_bolt_relative_angle': {'shape': [2500], 'var': 0.28346412082848743, 'finite': True}}`

## Next-step decision

- Control targets did not meet the pre-registered expectation. Inspect pipeline before escalating.
