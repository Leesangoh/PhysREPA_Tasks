# Scale Giant Verdict

## Scope

This document summarizes the completed `V-JEPA 2 Giant` scale-law run on:

- task: `push`
- feature recipe: `resid_post + temporal_last_patch`
- probe recipe: `token_patch + trainable 20-HP + 5-fold GroupKFold + zscore`
- targets:
  - `ee_direction_3d`
  - `ee_speed`

Completed seeds:

- `seed42`
- `seed123`

The originally planned `seed2024` run was intentionally skipped after the second successful Giant seed, because the scale-law bottleneck is now model coverage (`L/G/H`) rather than additional within-model seed averaging.


## Per-seed results

### `ee_direction_3d`

- `seed42`:
  - `L0 = 0.5211`
  - `L8 = 0.7669`
  - `peak = 0.8153 @ L24`
  - `last = 0.7983`
- `seed123`:
  - `L0 = 0.5528`
  - `L8 = 0.7762`
  - `peak = 0.8212 @ L30`
  - `last = 0.8081`

Aggregate:

- `L0 = 0.5369 ± 0.0159`
- `L8 = 0.7716 ± 0.0046`
- `peak = 0.8183 ± 0.0030`
- `peak layer = 27.0 ± 3.0`
- `last = 0.8032 ± 0.0049`

Interpretation:

- The Giant model preserves a strong direction-decoding regime.
- Compared with the Large baseline (`peak = 0.816 ± 0.001 @ L11.7 ± 0.9`), Giant maintains nearly the same peak `R^2` while shifting the peak substantially deeper.
- This is the first clear positive sign that the manipulation PEZ may obey a scale law in **layer location** more than in absolute peak magnitude.


### `ee_speed`

- `seed42`:
  - `L0 = 0.5175`
  - `L8 = 0.9134`
  - `peak = 0.9304 @ L25`
  - `last = 0.9179`
- `seed123`:
  - `L0 = 0.5137`
  - `L8 = 0.9131`
  - `peak = 0.9390 @ L25`
  - `last = 0.9092`

Aggregate:

- `L0 = 0.5156 ± 0.0019`
- `L8 = 0.9132 ± 0.0002`
- `peak = 0.9347 ± 0.0043`
- `peak layer = 25.0 ± 0.0`
- `last = 0.9136 ± 0.0043`

Interpretation:

- Speed remains highly decodable in Giant.
- As with direction, the dominant change from Large to Giant is not a collapse or explosion in `R^2`; it is a deeper peak location.


## Stability and diagnostics

Training scores remained near-saturated in both seeds:

- `ee_direction_3d`: train peak `~0.9999`
- `ee_speed`: train peak `~1.0000`

This means the Giant probe still has ample fitting capacity. The main scientific signal is the **validation-layer profile**, not trainability.


## Interim scale-law interpretation

Current evidence:

- `Large / ee_direction_3d`: `peak = 0.816 ± 0.001 @ L11.7 ± 0.9`
- `Giant / ee_direction_3d`: `peak = 0.818 ± 0.003 @ L27.0 ± 3.0`

So far, increasing model scale appears to:

1. preserve the existence of the manipulation-direction PEZ-like signal,
2. preserve peak magnitude, and
3. move the best-decoding layer substantially deeper.

This is exactly the scale-law pattern that motivates the `Huge` run.


## Next step

- delete the raw Giant token-patch cache,
- run `Huge`,
- then write the full `Large / Giant / Huge` scale-law verdict.
