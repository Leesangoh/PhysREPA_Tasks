# Phase 1 Results: push / large

## Summary

| target | output_dim | L0 | L8 | peak_r2 | peak_layer | last | class |
|---|---:|---:|---:|---:|---:|---:|---|
| mass | 1 | 0.0578 | 0.1068 | 0.2027 | 19 | 0.1502 | never-linear |
| obj_friction | 1 | 0.0063 | -0.0004 | 0.0376 | 18 | 0.0205 | never-linear |
| surface_friction | 1 | 0.0500 | 0.0525 | 0.0735 | 13 | 0.0575 | never-linear |
| ee_pos | 3 | 0.9611 | 0.9861 | 0.9925 | 19 | 0.9904 | always-linear |
| object_pos | 3 | 0.3722 | 0.4171 | 0.5152 | 5 | 0.0464 | PEZ-like |

## Hypothesis checks

- H1 (controls high from L0): `FAIL`
- H2 (static physics PEZ-like): `FAIL`

## Sanity checks

- feature_root: `/mnt/md1/solee/features/physprobe_vitl/push`
- n_episodes: `1500`
- num_layers: `24`
- feature_dim: `1024`
- window_count_mode: `58`
- missing_feature_keys: `0`
- groupkfold_overlap_zero: `True`
- gt_distribution_plot: `/home/solee/physrepa_tasks/artifacts/figures/gt_distribution_push.png`
- control_targets_l0_ge_0_8: `False`
- gt_summary: `{'mass': {'shape': [1500], 'var': 0.32154708209430666, 'finite': True}, 'obj_friction': {'shape': [1500], 'var': 0.14813523933536, 'finite': True}, 'surface_friction': {'shape': [1500], 'var': 0.22424630656730665, 'finite': True}, 'ee_pos': {'shape': [1500, 3], 'var': 0.06354377776806364, 'finite': True}, 'object_pos': {'shape': [1500, 3], 'var': 20.965989116026588, 'finite': True}}`

## Next-step decision

- Control targets did not meet the pre-registered expectation. Inspect pipeline before escalating.
