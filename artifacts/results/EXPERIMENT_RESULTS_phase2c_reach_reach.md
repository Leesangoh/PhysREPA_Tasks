# Phase 1 Results: reach / large

## Summary

| target | output_dim | L0 | L8 | peak_r2 | peak_layer | last | class |
|---|---:|---:|---:|---:|---:|---:|---|
| ee_direction_sincos | 2 | 0.3023 | 0.3532 | 0.3961 | 20 | 0.3951 | intermediate |
| ee_speed | 1 | 0.5981 | 0.8134 | 0.8253 | 17 | 0.7966 | intermediate |
| ee_accel_magnitude | 1 | 0.7168 | 0.8545 | 0.8734 | 10 | 0.8349 | PEZ-like |
| fake_mod5 | 1 | -0.5647 | -0.6726 | -0.4103 | 3 | -0.4696 | never-linear |

## Hypothesis checks

- H1 (controls high from L0): `FAIL`
- H2 (static physics PEZ-like): `FAIL`

## Sanity checks

- feature_root: `/mnt/md1/solee/features/physprobe_vitl_tokenpatch/reach`
- feature_type: `token_patch`
- run_tag: `phase2c_reach`
- n_episodes: `600`
- num_layers: `24`
- feature_dim_layer0: `262144`
- window_count_mode: `59`
- missing_feature_keys: `0`
- patch_shape: `[256, 1024]`
- groupkfold_overlap_zero: `True`
- gt_distribution_plot: `/home/solee/physrepa_tasks/artifacts/figures/gt_distribution_reach.png`
- control_targets_l0_ge_0_8: `False`
- gt_summary: `{'ee_direction_sincos': {'shape': [600, 2], 'var': 0.06826723811493109, 'finite': True}, 'ee_speed': {'shape': [600], 'var': 0.0017990203700788833, 'finite': True}, 'ee_accel_magnitude': {'shape': [600], 'var': 2.271594815793992, 'finite': True}, 'fake_mod5': {'shape': [600], 'var': 2.0, 'finite': True}}`

## Next-step decision

- Control targets did not meet the pre-registered expectation. Inspect pipeline before escalating.
