# Functional Significance Fusion Verdict

We reran the offline OOD action-chunk regression on the existing compact
Push/Drawer feature caches with one additional V-JEPA representation:

- `vjepa_pez = large:11`
- `vjepa_last = large:23`
- `vjepa_fusion = large:11+23`

This experiment tests a narrower claim than ``PEZ is the best control layer'':
whether PEZ-aligned features contribute complementary information when combined
with the late action layer.

## Push (mass/friction OOD)

- `vjepa_pez`: IID `0.9071 ± 0.0010`, OOD `0.8794 ± 0.0004`
- `vjepa_last`: IID `0.9193 ± 0.0002`, OOD `0.8916 ± 0.0005`
- `vjepa_fusion`: IID `0.9174 ± 0.0015`, OOD `0.8923 ± 0.0007`
- `videomae_best`: IID `0.9181 ± 0.0015`, OOD `0.8903 ± 0.0004`
- `dino_mid`: IID `0.8276 ± 0.0035`, OOD `0.7955 ± 0.0041`

Key comparison:
- `vjepa_fusion - vjepa_last` on OOD = `+0.00072`
- `vjepa_fusion - vjepa_pez` on OOD = `+0.01293`

Interpretation:
- On Push, fusion is the best OOD representation within the tested set, but the
  gain over `vjepa_last` is small.
- This does not support ``PEZ alone is best,'' but it does support the weaker
  complementary-information claim.

## Drawer (damping OOD)

- `videomae_best`: IID `0.9057 ± 0.0004`, OOD `0.9197 ± 0.0009`
- `vjepa_fusion`: IID `0.8968 ± 0.0006`, OOD `0.9125 ± 0.0002`
- `vjepa_last`: IID `0.8922 ± 0.0006`, OOD `0.9076 ± 0.0004`
- `vjepa_pez`: IID `0.8911 ± 0.0009`, OOD `0.9059 ± 0.0003`
- `dino_best`: IID `0.8735 ± 0.0000`, OOD `0.8893 ± 0.0007`

Key comparison:
- `vjepa_fusion - vjepa_last` on OOD = `+0.00486`
- `vjepa_fusion - vjepa_pez` on OOD = `+0.00659`

Interpretation:
- On Drawer, fusion produces a clear improvement over either single V-JEPA
  layer, but VideoMAE remains strongest overall.

## Bottom line

The final downstream claim is now stronger and more precise:

- `PEZ` is not the best standalone control layer.
- `late` features remain the strongest single V-JEPA layer for action
  prediction.
- But `PEZ + late` improves over either single V-JEPA layer on both Push and
  Drawer OOD control.

This supports the complementary-information version of the story:
PEZ-aligned representations are not merely analysis-friendly; they contribute
useful physics information when combined with the late action state.
