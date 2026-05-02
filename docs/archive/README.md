# `docs/archive/`

Archive of stale planning docs, post-mortem reports, and superseded experiment
plans. Files here are kept for provenance only — they are NOT load-bearing for
current work.

## Pre-2026-04-19 wave

Earlier autonomous-run logs and design docs:

- `NIGHTSHIFT_LOG.md`, `NIGHTSHIFT2_LOG.md`, `NIGHTSHIFT2_PROTOCOL.md`
- `PEZ_TO_PHYSPROBE_PLAN.md`, `EXPERIMENT_DESIGN.md`
- `CROSS_MODEL_PLAN.md`, `SCALE_PLAN.md`
- `F3_design.md`, `F5_design.md`, `R4_random_init_design.md`
- `ORAL_STRATEGY.md`, `260413_feedback.md`

## 2026-04-23 wave (post-methodology-incident)

Archived because the underlying numbers / assumptions were invalidated by the
2026-04-23 episode-mean probe-aggregation incident and the 2026-04-25 dataset
recollection:

- `REVISION_PLAN.md` — Layer 1 paper acceptance plan; the OOD R² values it
  hardens were computed with the contaminated episode-mean probe.
- `NIGHTSHIFT_FINAL_SUMMARY.md` — pre-incident summary of probing R² results
  (Push direction = 0.807, Strike direction = 0.885, etc.) — all episode-mean.
- `CONTACT_INFERENCE_ANALYSIS.md` — built around the assumption that the
  exported `contact_*` GT was zero-filled. The 2026-04-25 recollection
  restored nonzero contact forces, so the kinematic-surrogate proposal here
  is now redundant.

## Related

- `archive_old_wrong_probe/` (top of repo): the contaminated probe code itself
  + ~89 `EXPERIMENT_RESULTS_*.md` reports from that pipeline.
