# Phase 1 Results: push / large

## Summary

| target | output_dim | L0 | L8 | peak_r2 | peak_layer | last | class |
|---|---:|---:|---:|---:|---:|---:|---|
| ee_speed | 1 | 0.6711 | 0.9284 | 0.9312 | 13 | 0.9201 | PEZ-like |
| ee_accel_magnitude | 1 | 0.3468 | 0.6734 | 0.6943 | 14 | 0.6879 | PEZ-like |
| ee_direction | 1 | 0.0658 | 0.3816 | 0.4383 | 23 | 0.4383 | intermediate |
| object_speed | 1 | -0.2682 | 0.3301 | 0.3978 | 12 | 0.3814 | intermediate |
| object_accel_magnitude | 1 | 0.1995 | 0.5001 | 0.5224 | 20 | 0.5124 | intermediate |
| object_direction | 1 | -0.4051 | -0.0331 | 0.0287 | 23 | 0.0287 | never-linear |

## Hypothesis checks

- H1 (controls high from L0): `FAIL`
- H2 (static physics PEZ-like): `FAIL`

## Sanity checks

- feature_root: `/mnt/md1/solee/features/physprobe_vitl_tokenpatch/push`
- feature_type: `token_patch`
- run_tag: `phase2_kinematic`
- n_episodes: `1500`
- num_layers: `24`
- feature_dim_layer0: `262144`
- window_count_mode: `58`
- missing_feature_keys: `0`
- patch_shape: `[256, 1024]`
- groupkfold_overlap_zero: `True`
- gt_distribution_plot: `/home/solee/physrepa_tasks/artifacts/figures/gt_distribution_push.png`
- control_targets_l0_ge_0_8: `False`
- gt_summary: `{'ee_speed': {'shape': [1500], 'var': 0.0017155153710907913, 'finite': True}, 'ee_accel_magnitude': {'shape': [1500], 'var': 0.04132403452576831, 'finite': True}, 'ee_direction': {'shape': [1500], 'var': 0.3452866331070569, 'finite': True}, 'object_speed': {'shape': [1500], 'var': 0.0012873747148736104, 'finite': True}, 'object_accel_magnitude': {'shape': [1500], 'var': 0.20896853951153616, 'finite': True}, 'object_direction': {'shape': [1500], 'var': 1.5171538812340386, 'finite': True}}`

## Next-step decision

- Control targets did not meet the pre-registered expectation. Inspect pipeline before escalating.
