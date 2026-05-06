# PhysREPA Tasks — Isaac-Lab + Probing Repo

ICLR 2027 target. Active paradigm: **segment-level functional probing of
video / image SSL representations on manipulation videos**. See
`PhysProbe_Neurips_Paper/Sections/v2_segment_probing.tex` (Note 6) for the
paper-side state.

## Quick map

- `archive_data_collection/`: Isaac Lab data-collection code (misnamed but
  active). Per-task env configs, scripted policies, RL evaluation utilities.
  See `archive_data_collection/CLAUDE.md` for details.
- `probe/`: probing pipeline (V-JEPA / VideoMAE / DINOv2 / R3M).
  - `probe/scripts/`: feature extraction + target build + probing + analysis
  - `probe/utils/`: `dataset.py`, `targets.py`, `functionals.py`, `io.py`, `probe.py`, `vjepa_loader.py`, `stats.py`
  - `probe/results/`: per-task probe CSVs + heatmaps + manifest
  - `probe/cache/`: per-episode cached targets (small) — feature caches symlinked to `/mnt/md1/solee/physprobe_features/`
- `PhysProbe_Neurips_Paper/`: paper TeX (gitignored; PDFs copied to `paper_pdf/`)
- `EXPERIMENTS_SUMMARY.md`: human-friendly summary of all experiments
- `paper_pdf/`: compiled PDFs at milestone snapshots

## What's where (current)

| Need | File |
|------|------|
| Run a new probe sweep | `probe/scripts/03_run_probe.py --task <T> --variant <V> --target-set <S>` |
| Build segment-level functional targets | `probe/scripts/02b_build_functional_targets.py` |
| Extract features for new model | `probe/scripts/01b_extract_cross_model_features.py` (VideoMAE, DINOv2) |
| Generate paper main figure (heatmap) | `probe/scripts/20_functional_pez_heatmap.py` |
| What experiments map to what scripts | `probe/results/MANIFEST.md` |
| What can be safely freed from disk | `probe/results/CACHE_MANIFEST.md` |

## Reproducibility — minimum to reproduce headline claim

See `probe/results/MANIFEST.md` "Reproducibility runbook" section.

Headline claim to reproduce:
> Video SSL pretraining (V-JEPA, VideoMAE) selectively linearizes
> higher-order dynamics summaries; image SSL (DINOv2) saturates lower.
> Acceleration peak\_to\_peak gap on push: V-JEPA 0.802, VideoMAE 0.774,
> DINOv2 0.451 (gap −0.35).

Reproducing requires:
1. The recollected dataset at `/home/solee/data/data/isaac_physrepa_v2_recollected_2026-04-23/`
2. Local model weights at `/mnt/md1/solee/checkpoints/`
3. `/isaac-sim/python.sh` interpreter (Isaac Sim 4.5.0, torch 2.7.0+cu128)
4. The runbook commands in `probe/results/MANIFEST.md`

## Current status (2026-05-06)

- Paper Note 6 (28-page PDF at `paper_pdf/PhysProbe_2026-05-06.pdf`)
- 12 probe sweeps complete (V-JEPA A 6/6, DINOv2 A 5/6 + partial nut, V-JEPA B push, VideoMAE A push, VideoMAE B push)
- 3 GPU autonomous chains running
- Disk: 261G free on `/mnt/md1/solee` (3.6T total, 93% used)
- Pending experiments: D (DINOv2 nut retry), H (bootstrap CI), G (F5 functional), F (F4-A controls), C/B subsets, E (ViT-G), I (classifier head)

See `/root/.claude/plans/iclr27_autonomous_plan.md` for the full execution plan.

## Quarantined / archived

- `archive_old_wrong_probe/`: pre-2026-04-23 probe code (episode-mean bug). Provenance only.
- `docs/archive/`: superseded planning docs.

## Repo notes

- `PhysProbe_Neurips_Paper/` is gitignored (TeX nested git, secrets in `token.txt`)
- Compiled PDFs go to `paper_pdf/<name>.pdf` for sharing
- New collection code under `archive_data_collection/` (re-export shims at top level)
- Strict no-delete policy on running probe caches (see `probe/results/CACHE_MANIFEST.md`)
