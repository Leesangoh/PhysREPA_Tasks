# Surrogate Validation Verdict

## Result

Native Strike `contact_force` validation is **possible** on the recollected export.

The recollected dataset contains nonzero native `contact_force` and `contact_flag` windows, so surrogate-vs-native alignment can be measured directly.

## Full Strike Audit

- audited episodes: `1000`
- native `contact_force` nonzero episodes: `966` / `1000`
- native `contact_finger_l_object_force` nonzero episodes: `966` / `1000`
- native `contact_object_surface_force` nonzero episodes: `988` / `1000`
- native `contact_flag` nonzero episodes: `957` / `1000`
- max exported native `contact_force` magnitude over all audited frames: `192.213479`
- max exported native `contact_flag` over all audited frames: `1.000000`
- max object-acceleration magnitude over the same audit: `240.948192`

## Matched Window-Level Check

- audited windows: `44982`
- native `contact_force` nonzero windows: `6686`
- native `contact_flag` nonzero windows: `6591`
- surrogate force-proxy max over those windows: `240.948192`
- native `contact_force` max over those windows: `192.213479`
- Pearson surrogate/native correlation: `0.4175709297717998`
- Spearman surrogate/native correlation: `0.5827175063843649`
- top-100 window overlap fraction: `0.08`


## Consequence for the Paper

- Native `contact_force` is present in the recollected Strike export.
- Surrogate/native alignment can now be measured directly on matched windows.
- Native-force probing is scientifically valid on this recollected dataset.
