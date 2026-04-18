# Phase 2 Results: Push / Large / Token-Patch

## Verdict

- Overall Phase 2 verdict: `Scenario A partial`
- Interpretation:
  - the Push token-patch PEZ recipe does recover a paper-like emergence pattern on the **ee-side direction** target,
  - but the same is **not** true for object-side direction or static physics parameters.

## Fixed recipe

- feature cache: `token_patch`
- capture: `resid_post`
- pooling: `temporal_last_patch`
- feature root: `/mnt/md1/solee/features/physprobe_vitl_tokenpatch/push`
- model: `V-JEPA 2 Large`
- probe: `trainable 20-HP sweep`
- CV: `5-fold GroupKFold by episode_id`
- norm: `zscore`

## Hypothesis evaluation

### H1 — Robot-state control should be linearly decodable from shallow layers

- `ee_pos`: `PASS`
  - `L0 = 0.9440`, peak `0.9625 @ L12`
- `ee_speed`: `PASS`
  - `L0 = 0.6711`, peak `0.9312 @ L13`
- `ee_accel_magnitude`: `partial`
  - `L0 = 0.3468`, peak `0.6943 @ L14`

Conclusion:
- The arm-side control family is clearly decodable.
- This argues against a broken token-patch pipeline.

### H2 — A PEZ-like intermediate-depth emergence should appear on at least one manipulation-physics analog

- `ee_direction` with scalar angle: `FAIL`
  - `L0 = 0.0658`, `L8 = 0.3816`, peak `0.4383 @ L23`
- `ee_direction_sincos`: `PASS`
  - `L0 = 0.5554`, `L8 = 0.7894`, peak `0.8068 @ L13`, `L23 = 0.7969`
- `object_direction` with scalar angle: `FAIL`
  - peak `0.0287 @ L23`
- `object_direction_sincos`: `FAIL`
  - peak `0.0841 @ L23`

Conclusion:
- The PEZ-like pattern is recovered **only** for `ee_direction_sincos`.
- Circular wrap is therefore a real issue for manipulation direction targets.
- The object-side direction signal does not show the same emergence.

## Summary table

| family | target | parameterization | L0 | L8 | peak_r2 | peak_layer | last | judgment |
|---|---|---|---:|---:|---:|---:|---:|---|
| static physics | mass | scalar | -0.1451 | 0.0239 | 0.0991 | 14 | 0.0728 | never-linear |
| static physics | obj_friction | scalar | -0.2647 | -0.1305 | -0.0626 | 3 | -0.1389 | never-linear |
| static physics | surface_friction | scalar | -0.2270 | -0.0895 | -0.0387 | 22 | -0.0422 | never-linear |
| control | ee_pos | vector | 0.9440 | 0.9593 | 0.9625 | 12 | 0.9508 | always-linear |
| control | object_pos | vector | 0.6980 | 0.6772 | 0.7177 | 2 | 0.4200 | intermediate |
| kinematic | ee_speed | scalar | 0.6711 | 0.9284 | 0.9312 | 13 | 0.9201 | always-linear |
| kinematic | ee_accel_magnitude | scalar | 0.3468 | 0.6734 | 0.6943 | 14 | 0.6879 | intermediate |
| kinematic | ee_direction | angle | 0.0658 | 0.3816 | 0.4383 | 23 | 0.4383 | fail |
| kinematic | ee_direction | sincos | 0.5554 | 0.7894 | 0.8068 | 13 | 0.7969 | PEZ-like |
| kinematic | object_speed | scalar | -0.2682 | 0.3301 | 0.3978 | 12 | 0.3814 | intermediate |
| kinematic | object_accel_magnitude | scalar | 0.1995 | 0.5001 | 0.5224 | 20 | 0.5124 | intermediate |
| kinematic | object_direction | angle | -0.4051 | -0.0331 | 0.0287 | 23 | 0.0287 | fail |
| kinematic | object_direction | sincos | -0.2202 | 0.0483 | 0.0841 | 23 | 0.0841 | fail |

## Sanity checks

- token cache shape verified:
  - `layer_0_window_0.shape = (256, 1024)`
  - `window_count_mode = 58`
- train/val episode overlap:
  - `0`
- target variance:
  - non-zero for every Phase 2 target
- key control:
  - `ee_pos L0 = 0.9440`

## Phase 1 -> Phase 2 comparison

- `mass`
  - Phase 1 mean-pool peak: `0.203 @ L19`
  - Phase 2 token-patch peak: `0.099 @ L14`
  - conclusion: token-patch did **not** rescue static mass
- `ee-side direction`
  - scalar angle failed
  - `sin/cos` succeeded with a PEZ-like mid-depth peak
- `object-side direction`
  - failed under both angle and `sin/cos`

## Final interpretation

- The Push task does **not** reproduce the full single-ball PEZ story.
- Static physics parameters remain effectively non-linear-decodable in Phase 2.
- The strongest positive result is:
  - `ee_direction_sincos` peak `0.8068 @ L13`
  - with high kinematic controls (`ee_pos`, `ee_speed`) and a mild late decline (`0.8068 -> 0.7969`)
- This supports a **partial manipulation-domain PEZ analog**:
  - the arm-side motion direction emerges with a PEZ-like shape under the best token-patch recipe,
  - but the object-side interaction signal does not match that pattern yet.

## Next step

- Proceed to Phase 3 event-aligned force probing:
  - `contact_flag`
  - `contact_force_magnitude`
- Use window-level targets with episode-grouped CV.
