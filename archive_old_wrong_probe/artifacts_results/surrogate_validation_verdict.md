# Surrogate Validation Verdict

## Result

Native Strike `contact_force` validation is **not possible** with the public Step 0 export.

The reason is empirical, not inferential: the exported native `contact_force`,
`contact_finger_l_object_force`, `contact_object_surface_force`, and `contact_flag`
channels are zero-filled across the audited Strike data, while object-acceleration
spikes remain large and frequent.

## Full Strike Audit

- audited episodes: `3000`
- native `contact_force` nonzero episodes: `0` / `3000`
- native `contact_finger_l_object_force` nonzero episodes: `0` / `3000`
- native `contact_object_surface_force` nonzero episodes: `0` / `3000`
- native `contact_flag` nonzero episodes: `0` / `3000`
- max exported native `contact_force` magnitude over all audited frames: `0.000000`
- max exported native `contact_flag` over all audited frames: `0.000000`
- max object-acceleration magnitude over the same audit: `279.350533`

## Matched Window-Level Check

- audited windows: `45208`
- native `contact_force` nonzero windows: `0`
- native `contact_flag` nonzero windows: `0`
- surrogate force-proxy max over those windows: `157.296943`
- native `contact_force` max over those windows: `0.000000`
- Pearson surrogate/native correlation: `None`
- Spearman surrogate/native correlation: `None`
- top-100 window overlap fraction: `None`

Because the native target has zero variance, correlation and rank-consistency are
undefined rather than merely weak.

## Consequence for the Paper

- The current surrogate-contact analysis remains necessary.
- We cannot run a meaningful native `contact_force` probe ranking with the public Step 0 Strike export.
- The scientifically correct update is therefore a stronger data audit: native contact channels were rechecked and remain zero-filled, so the Tier-B claim still rests on a surrogate force proxy rather than simulator-native force supervision.
