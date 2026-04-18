# PhysREPA Tasks: PEZ Probing Focus

This repository is currently organized around one goal:

- applying the PEZ probing methodology from the PEZ reproduction project to PhysProbe manipulation data

The active planning document is:

- [PEZ_TO_PHYSPROBE_PLAN.md](./PEZ_TO_PHYSPROBE_PLAN.md)

## Current focus

The next planned work is:

1. write a new probing script for manipulation data
2. load frozen ViT-L / ViT-G PhysProbe features from:
   - `/mnt/md1/solee/features/physprobe_vitl/`
   - `/mnt/md1/solee/features/physprobe_vitg/`
3. load ground-truth physics targets from the PhysProbe parquet dataset
4. run PEZ-style layer-wise probing and classify each variable as:
   - PEZ-like
   - always decodable
   - never decodable

Planned outputs will go under:

- `./artifacts/`

## Historical note

Older code for:

- data collection
- RL training
- Isaac Lab environment configuration
- rollout / recording
- verification
- older probing experiments

has been preserved under:

- `./archive_data_collection/`

That archive is kept for record-keeping and recovery, but it is not the active working surface for the PEZ-to-PhysProbe probing effort.

## Immediate next step

Implement:

- `probe_physprobe.py`

based on:

- the plan in [PEZ_TO_PHYSPROBE_PLAN.md](./PEZ_TO_PHYSPROBE_PLAN.md)
- the PEZ probing logic from `/home/solee/pez/step3_probe.py`

