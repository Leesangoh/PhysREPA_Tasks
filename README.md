# PhysREPA Tasks

This repository contains the experiment code, analysis scripts, and result
artifacts used to study manipulation physics representations in video world
models.

## Primary documents

- [NIGHTSHIFT_FINAL_SUMMARY.md](./NIGHTSHIFT_FINAL_SUMMARY.md): high-level
  summary of the major experimental phases and conclusions
- [CONTACT_INFERENCE_ANALYSIS.md](./CONTACT_INFERENCE_ANALYSIS.md): contact-label
  audit and surrogate-contact analysis notes

## Repository layout

- `artifacts/results/`: canonical CSV, JSON, and markdown verdict files used by
  the paper and figure-generation scripts
- `artifacts/figures/`: experiment-side plots and diagnostics
- `archive_data_collection/`: older Isaac Lab collection and training code kept
  for reference and recovery
- `docs/archive/`: historical planning documents, run protocols, and nightshift
  logs preserved for provenance

## Current scope

The finalized analysis covers:

- V-JEPA scale comparisons (`L/G/H`)
- cross-model comparisons against VideoMAE-L and DINOv2-L
- native contact-force recollection and Tier-B validation
- functional OOD action regression on Push and Drawer
- selective scope expansion to PegInsert and NutThread

The active paper source lives in a separate nested git repository at
`PhysProbe_Neurips_Paper/`, which is intentionally ignored from this repo's git
status.
