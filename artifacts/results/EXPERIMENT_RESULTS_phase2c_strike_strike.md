# Phase 1 Results: strike / large

## Summary

| target | output_dim | L0 | L8 | peak_r2 | peak_layer | last | class |
|---|---:|---:|---:|---:|---:|---:|---|
| ee_direction_sincos | 2 | 0.6974 | 0.8707 | 0.8855 | 11 | 0.8773 | PEZ-like |
| ee_speed | 1 | 0.8832 | 0.9572 | 0.9630 | 11 | 0.9513 | always-linear |
| ee_accel_magnitude | 1 | 0.6638 | 0.8759 | 0.8963 | 12 | 0.8783 | PEZ-like |
| fake_mod5 | 1 | -0.4617 | -0.3494 | -0.2896 | 12 | -0.3062 | never-linear |

## Hypothesis checks

- H1 (controls high from L0): `FAIL`
- H2 (static physics PEZ-like): `FAIL`

## Sanity checks

- feature_root: `/mnt/md1/solee/features/physprobe_vitl_tokenpatch/strike`
- feature_type: `token_patch`
- run_tag: `phase2c_strike`
- n_episodes: `2895`
- num_layers: `24`
- feature_dim_layer0: `262144`
- window_count_mode: `58`
- missing_feature_keys: `0`
- patch_shape: `[256, 1024]`
- groupkfold_overlap_zero: `True`
- gt_distribution_plot: `/home/solee/physrepa_tasks/artifacts/figures/gt_distribution_strike.png`
- control_targets_l0_ge_0_8: `False`
- gt_summary: `{'ee_direction_sincos': {'shape': [2895, 2], 'var': 0.05700854434204595, 'finite': True}, 'ee_speed': {'shape': [2895], 'var': 0.007827433056543546, 'finite': True}, 'ee_accel_magnitude': {'shape': [2895], 'var': 0.16442947580164646, 'finite': True}, 'fake_mod5': {'shape': [2895], 'var': 2.0, 'finite': True}}`

## Next-step decision

- Control targets did not meet the pre-registered expectation. Inspect pipeline before escalating.
