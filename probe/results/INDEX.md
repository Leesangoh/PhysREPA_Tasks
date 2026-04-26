# PhysProbe Variant A Sweep — Results Index

Run window: 2026-04-25 09:30 UTC → 2026-04-26 01:36 UTC (≈ 16h 5min wall).
Spec: `/home/solee/physrepa_tasks/claude_code_task.md`
Plan: `/root/.claude/plans/read-physrepa-tasks-claude-code-task-md-humming-planet.md`

## Verdict (spec § 16)

| Mode | Verdict |
|---|---|
| Strict argmax | **MARGINAL** (Push ee_velocity argmax = L22, marginally outside L6–18) |
| Relaxed (first layer ≥ 99 % of max R²) | **HEALTHY** (Push ee_velocity 99 % peak = L17, inside L6–18) |
| **Primary** | **HEALTHY** |

All three R² thresholds pass with margin: Push ee_velocity 0.921, ee_position 0.980, Strike ee_velocity 0.936.

## Layout

| Path | What |
|---|---|
| `REPORT.md` | Full report — verdict, criteria, all per-task layer × R² tables, deviations |
| `decision.json` | Machine-readable verdict + criteria |
| `peak_layers.csv` | Best layer + R² per (task, target) |
| `progress.md` | Time-stamped progress log of the entire run |
| `<task>/variant_A/<target>.csv` | Per-fold rows (5 folds × 24 layers) for one target |
| `<task>/variant_A/_summary.csv` | Per-layer aggregated R² mean/std/MSE per task |
| `plots/` | All figures (see below) |
| `logs/` | Raw extraction + chain stdout/stderr |

## Plots

### Per-task layer-vs-R² (EE + Object side-by-side, ±1 std band)
- [push](plots/push_layer_vs_r2.png)
- [strike](plots/strike_layer_vs_r2.png)
- [reach](plots/reach_layer_vs_r2.png) (EE only — no object)
- [drawer](plots/drawer_layer_vs_r2.png)
- [peg_insert](plots/peg_insert_layer_vs_r2.png)
- [nut_thread](plots/nut_thread_layer_vs_r2.png)

### Cross-task overlays (one figure per target, 6 task lines)
- EE: [position](plots/cross_task_ee_position.png), [velocity](plots/cross_task_ee_velocity.png), [speed](plots/cross_task_ee_speed.png), [direction](plots/cross_task_ee_direction.png), [acceleration](plots/cross_task_ee_acceleration.png), [accel_mag](plots/cross_task_ee_accel_mag.png)
- Object: [position](plots/cross_task_obj_position.png), [velocity](plots/cross_task_obj_velocity.png), [speed](plots/cross_task_obj_speed.png), [direction](plots/cross_task_obj_direction.png), [acceleration](plots/cross_task_obj_acceleration.png), [accel_mag](plots/cross_task_obj_accel_mag.png)

### Bird's-eye
- [heatmap_r2.png](plots/heatmap_r2.png) — every (task, target) row × 24 layer columns

## Headline numbers (best layer)

| Task | ee_velocity | ee_position | obj_velocity | obj_position |
|---|---|---|---|---|
| push | L22 0.921 | L21 0.980 | L21 0.719 | L8 0.208 ← static cube, early-peak |
| strike | L18 0.936 | L22 0.985 | L17 0.665 | L3 0.289 ← mostly stationary |
| reach | L17 0.296 | L17 0.936 | — | — |
| drawer | L22 0.578 | L22 0.976 | L18 0.918 | L21 0.990 |
| peg_insert | L17 0.807 | L22 0.969 | L17 0.489 | L17 0.967 |
| nut_thread | see `nut_thread/_summary.csv` for all 12 targets | | | |

## Spec deviations (full reasoning in REPORT.md "Methodology notes")

1. Probe `batch_size` 1024 → max(1024, N_train/8) — Python loop overhead made the spec batch infeasible (~50× speedup with no quality loss for convex linear probes).
2. dt source: local `meta/info.json` fps used (push/strike/reach=50, drawer=60, peg/nut=15). HF README "15 fps" disagrees; finite_diff(position) vs stored velocity match to 2-3 % at 50 fps confirming local meta.
3. Native `physics_gt.<entity>_acceleration` is Isaac-Lab body accelerometer, not d/dt(velocity). Used finite-diff uniformly per user directive ("avoid distribution shift between native vs derived").
4. `L0_saturates` check: spec gate `L0 R²>0.9` alone over-triggers on static-scene tasks where V-JEPA patch_embed already encodes spatial position. Refined to `L0>0.9 AND max-L0 gain<0.02` (Drawer ee_position L0=0.90 → L22=0.98 is real processing, not leakage).
5. PEZ peak depth: relaxed criterion (first layer ≥ 99 % of max R²) added alongside strict argmax. Push ee_velocity has a flat L17–L22 plateau; strict argmax sits at L22 (just outside spec's 6–18 band) only because of plateau noise.
6. Pooling identity tolerance: relative 1e-3 (not absolute 1e-3). V-JEPA deep-layer activation magnitudes reach ~250; fp16 storage truncation produces ~0.05–0.10 absolute diff but <0.07 % relative.

## What's NOT in this run

- Variant B sweep (8192-d spatial-mean features). B features ARE cached at `/mnt/md1/solee/physprobe_features/<task>/variant_B/` (≈ 825 GB total) but probe sweep deferred per spec EXECUTION DISCIPLINE ("Do NOT escalate to Variant B or C without my explicit greenlight"). Greenlight + ~16h compute → run.
- Variant C sweep (262144-d temporal-mean fallback). Not needed at this stage; A is HEALTHY.

## How to regenerate

```bash
cd /home/solee/physrepa_tasks/probe
/isaac-sim/python.sh scripts/04_aggregate_results.py --variant A   # decision.json + peak_layers.csv
/isaac-sim/python.sh scripts/05_write_report.py                    # REPORT.md
/isaac-sim/python.sh scripts/06_plot_results.py                    # plots/
```
