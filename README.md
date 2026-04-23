# PhysREPA Tasks

This repository is now centered on **dataset collection, value auditing, and
dataset-release recovery** for PhysREPA Step 0.

After the methodology incident documented outside the repo, the old probing and
paper-analysis stack was removed from the active tree and archived for
provenance under `archive_old_wrong_probe/`.

## Primary documents

- [NIGHTSHIFT_FINAL_SUMMARY.md](./NIGHTSHIFT_FINAL_SUMMARY.md): historical
  summary of the earlier experiment phases
- [CONTACT_INFERENCE_ANALYSIS.md](./CONTACT_INFERENCE_ANALYSIS.md): contact-label
  audit notes from the previous analysis phase
- [REVISION_PLAN.md](./REVISION_PLAN.md): latest user-provided revision plan

## Active layout

- `archive_data_collection/`: Isaac Lab collection code and env definitions
- `envs/`, `mdp/`, `policies/`: import shims used by the collection code
- `artifacts/results/`: current audit summaries kept active
- `artifacts/figures/`: current audit visuals kept active
- `artifacts/notebooks/`: small audit helpers

## Archived layout

- `archive_old_wrong_probe/scripts/`: old probe, feature-extraction, and
  analysis scripts
- `archive_old_wrong_probe/artifacts_results/`: old result CSV/JSON/markdown
- `archive_old_wrong_probe/artifacts_figures/`: old probe/result figures
- `archive_old_wrong_probe/artifacts_logs/`: old run logs
- `docs/archive/`: historical planning docs and nightshift logs

## Notes

- The nested paper repo `PhysProbe_Neurips_Paper/` remains intentionally
  ignored from this repo's git status.
- Current active work should stay scoped to dataset regeneration and value
  validation unless explicitly redirected.
