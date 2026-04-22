# Phase 1 Results: peg_insert / large

## Summary

| target | output_dim | L0 | L8 | peak_r2 | peak_layer | last | class |
|---|---:|---:|---:|---:|---:|---:|---|
| ee_direction_3d | 3 | 0.1577 | 0.4164 | 0.5543 | 20 | 0.5381 | intermediate |

## Hypothesis checks

- H1 (controls high from L0): `FAIL`
- H2 (static physics PEZ-like): `FAIL`

## Sanity checks

- feature_root: `/mnt/md1/solee/features/physprobe_vitl/peg_insert`
- feature_type: `mean`
- run_tag: `scope_peg_insert_multiseed_seed123`
- n_episodes: `2500`
- num_layers: `24`
- feature_dim_layer0: `1024`
- window_count_mode: `33`
- missing_feature_keys: `0`
- patch_shape: `None`
- groupkfold_overlap_zero: `True`
- gt_distribution_plot: `/home/solee/physrepa_tasks/artifacts/figures/gt_distribution_peg_insert.png`
- control_targets_l0_ge_0_8: `False`
- gt_summary: `{'ee_direction_3d': {'shape': [2500, 3], 'var': 0.018379380977733572, 'finite': True}}`

## Next-step decision

- Control targets did not meet the pre-registered expectation. Inspect pipeline before escalating.
