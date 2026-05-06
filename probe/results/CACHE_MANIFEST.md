# Cache Manifest — Disk Tracking & Cleanup Log

Per Codex disk strategy: incremental task-boundary cleanup only after all
downstream artifacts (CSV, summary, bootstrap CI, heatmap) are locked.

## Live cache directories (no cleanup until task completes)

- Active probes:
  - GPU 0: V-JEPA Variant B strike → keep `variant_B/strike` data + features
  - GPU 1: VideoMAE-L Variant A drawer / strike / reach / peg / nut → keep
  - GPU 3: VideoMAE-L Variant B drawer / ... → keep

## NEVER delete

- V-JEPA Variant A baseline (all 6 tasks) — `physprobe_features/<task>/variant_A/`
- V-JEPA push/strike shuffled cache — needed for functional F5 (Task G)
  `physprobe_features/{push,strike}/variant_A_shuffled/`
- canonical target files — `probe/cache/<task>/targets*.npz`
- new functional/effect target files — `probe/cache/<task>/targets_functional_*.npz`, `targets_effect.npz`

## Safe cleanup ORDER (per Codex)

Top of list = safest to delete first.

| Step | Cache | Size est | Free condition |
|------|-------|---------:|----------------|
| 1 | `<task>/variant_B_videomae_large/` | ~125G push, ~250G strike, ~190G drawer | per task: probe CSV done + 5-model heatmap reflects + B≈A claim CI locked |
| 2 | `<task>/variant_B/` for reach, peg_insert, nut_thread (V-JEPA VarB non-headline) | ~52G + ~119G + ~119G | per task: probe CSV done if computed; otherwise drop sweep |
| 3 | `<task>/variant_A_dinov2_large/` | ~16G push, ~25G strike, ~26G drawer, ~6G reach, ~15G peg/nut each | per task: 5-task gap matrix locked + heatmap reflects + CI locked |
| 4 | `<task>/variant_A_videomae_large/` | ~16G push, ~25G strike, ~26G drawer, ~6G reach, ~15G peg/nut each | per task: 5-model figure locked (last to delete) |
| 5 | `<task>/r3m/` (Note H R3M baseline) | ~8G total | If no further analysis planned |
| 6 | `<task>/variant_B/` for push/strike/drawer (V-JEPA VarB headline) | ~125G + ~197G + ~206G | A≈B claim figure/CI locked (last to delete) |

## Free target on each cleanup pass

- Goal: keep ≥ 200G free at all times.
- Hard floor: 140G — stop new launches if approaching this.

## Cleanup audit log

(append entries as cleanup happens)

- *(no cleanup yet)*

## Disk snapshot baseline (2026-05-06 16:00 UTC)

```
/dev/md1                          ext4     3.6T  3.3T  261G  93% /mnt/md1/solee
```

Top-level breakdown:
```
1.3T+  /mnt/md1/solee/physprobe_features/   <-- target of cleanup
 818G  V-JEPA Variant B (6 tasks)
 818G+ VideoMAE Variant B (push complete; strike/drawer growing)
 ~120G V-JEPA Variant A (6 tasks)
  92G  VideoMAE Variant A (3 of 6 done)
 ~80G  DINOv2 Variant A (5 of 6 done)
  41G  V-JEPA Variant A shuffled (push, strike) — keep for functional F5
   8G  R3M
```

(Actual numbers refresh at each cleanup pass.)
