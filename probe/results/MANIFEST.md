# PhysProbe Experiment Manifest

Maps each script to its inputs, outputs, paper section, and status.

## V-JEPA / VideoMAE / DINOv2 feature extraction

| Script | Model | Inputs | Outputs | Paper | Status |
|--------|-------|--------|---------|-------|--------|
| `01_extract_features.py` | V-JEPA 2 ViT-L | recollected dataset MP4 | `<task>/variant_{A,B}/episode_*.npz` | base | ✅ 6 tasks A+B |
| `01_extract_features.py --shuffle-frames` | V-JEPA 2 ViT-L | same | `<task>/variant_A_shuffled/` | F5 | ✅ push, strike |
| `01b_extract_cross_model_features.py --model videomae_large` | VideoMAE-L | same | `<task>/variant_A_videomae_large/`, `variant_B_videomae_large/` | Note 6 cross-model | ✅ 6 tasks A+B |
| `01b_extract_cross_model_features.py --model dinov2_large --variant-a-only` | DINOv2-L | same | `<task>/variant_A_dinov2_large/` | Note 6 cross-model | ✅ 6 tasks A |
| `11_extract_r3m_features.py` | R3M ResNet50 | same | `<task>/r3m/` | Appendix Note H | ✅ push, strike, drawer |

## Target builders

| Script | Inputs | Outputs | Paper |
|--------|--------|---------|-------|
| `02_build_targets.py` | parquet trajectories | `cache/<task>/targets.npz` | base |
| `02b_build_functional_targets.py` | parquet | `cache/<task>/targets_functional_{short,med,long,prefix}.npz` | Note 6 |
| `02c_build_effect_targets.py` | parquet | `cache/<task>/targets_effect.npz` | Note 6 (TODO classifier) |

## Probing

| Script | Variant | Target-set | Outputs | Paper |
|--------|---------|-----------|---------|-------|
| `03_run_probe.py` | A / B / A_videomae_large / B_videomae_large / A_dinov2_large | default / functional_short / functional_* / effect | `<task>/<variant>/<target>.csv` | Notes 1-6 |
| `08_time_only_baseline.py` | n/a | default | `time_only_baseline.csv` | leakage check |
| `09_physics_condition_split.py` | A | default | `physics_condition_split/<task>__<param>.csv` | F4-A |
| `12_run_r3m_probe.py` | r3m | default | `<task>/r3m/<target>.csv` | Note H |

## Aggregation / analysis

| Script | Inputs | Outputs | Paper |
|--------|--------|---------|-------|
| `04_aggregate_results.py` | per-task variant CSVs | `<task>/<variant>/_summary.csv` | all |
| `10_bootstrap_cis.py` | per-fold sidecar CSVs | `bootstrap_cis.csv` (3120 rows for prior round) | F5/F4-A/F2/transfer |
| `20_functional_pez_heatmap.py` | per-task variant functional CSVs | `figures/functional_pez_heatmap_*.pdf`, summary CSV | Note 6 main figure |

## Trajectory analysis

| Script | Outputs | Paper |
|--------|---------|-------|
| `trajectory_analysis_B/scripts/15_temporal_order_ablation.py` | Phase D | Note 5 |
| `trajectory_analysis_B/scripts/16_cross_task_transfer.py` | Phase E | Note 5 |
| `trajectory_analysis_B/scripts/17_phase_conditional.py` | F2 | Note 4 |
| `trajectory_analysis_B/scripts/18_phase_space_geometry.py` | F1-b | Note 5 |

## Status (2026-05-06 16:30 UTC)

### Completed probe sweeps (12)
- V-JEPA Variant A: push, strike, drawer, reach, peg_insert, nut_thread (6/6)
- DINOv2 A: push, drawer, strike, reach, peg_insert (5/6 — nut OOM partial 41%)
- V-JEPA Variant B: push (95%, L0-L21 saved)
- VideoMAE A: push
- VideoMAE B: push

### In progress (3 GPU)
- GPU 0: V-JEPA VarB strike
- GPU 1: VideoMAE A drawer (chain → strike → reach → peg → nut)
- GPU 3: VideoMAE B drawer (chain → strike → ...)

### Pending tasks (Codex priority A→D→H→J→G→F→C(subset)→B(subset)→E→I)
- **D** DINOv2 nut retry
- **H** Bootstrap CI on new functional claims
- **J** 6×5 functional-PEZ heatmap (script ✅, runs incrementally)
- **G** F5 frame-shuffle on functional targets
- **F** F4-A disambiguation on functional targets
- **C** V-JEPA Variant B headline subset continuation
- **B** VideoMAE Variant B headline subset continuation
- **E** ViT-G scale sweep (deferred — disk constraint)
- **I** classifier head for contact_involvement (default drop)

## Reproducibility runbook (minimum to reproduce headline claim)

```bash
# 0. Setup
cd /home/solee/physrepa_tasks
PYTHON=/isaac-sim/python.sh

# 1. V-JEPA Variant A baseline (already done — feature cache lives at /mnt/md1)
$PYTHON probe/scripts/01_extract_features.py --task push --gpu 0 --batch 8 --shards 1 --shard-id 0

# 2. Build per-task functional targets (CPU-only, ~5 min)
$PYTHON probe/scripts/02b_build_functional_targets.py --task push --scale short

# 3. Run V-JEPA Variant A functional probe (GPU, ~6h per task)
$PYTHON probe/scripts/03_run_probe.py --task push --variant A --gpu 0 \
  --target-set functional_short --targets all --layers all

# 4. Cross-model: extract features (GPU, ~5h per task per model)
$PYTHON probe/scripts/01b_extract_cross_model_features.py \
  --model videomae_large --task push --gpu 0 --batch 4
$PYTHON probe/scripts/01b_extract_cross_model_features.py \
  --model dinov2_large --task push --gpu 0 --batch 8 --variant-a-only

# 5. Cross-model probe sweep
$PYTHON probe/scripts/03_run_probe.py --task push --variant A_videomae_large --gpu 0 \
  --target-set functional_short --targets all --layers all --per-layer-load
$PYTHON probe/scripts/03_run_probe.py --task push --variant A_dinov2_large --gpu 0 \
  --target-set functional_short --targets all --layers all --per-layer-load

# 6. Heatmap & summary (CPU)
$PYTHON probe/scripts/20_functional_pez_heatmap.py
```

Outputs land in `probe/results/<task>/<variant>/<target>.csv` and
`probe/results/figures/functional_pez_heatmap_*.pdf`.
