## Round 4 Verdict: Pretraining vs. Architecture-Only on Push

This round holds the architecture and probe recipe fixed and changes only the
backbone weights. We compare the pretrained V-JEPA 2 Large Push baseline
against the same V-JEPA 2 Large architecture with random initialization
(`model_seed=0`). Both conditions use the same token-patch recipe:
`resid_post + temporal_last_patch + flatten + zscore + 5-fold GroupKFold by episode_id + trainable 20-HP sweep`.

### Main result

Random initialization does **not** reproduce the pretrained mid-depth kinematic
regime.

| Target | Model | L0 | L8 | Peak R² | Peak layer | Last |
|---|---|---:|---:|---:|---:|---:|
| `ee_direction_3d` | pretrained | `0.648 ± 0.007` | `0.804 ± 0.002` | `0.816 ± 0.001` | `11.7 ± 0.9` | `0.811 ± 0.002` |
| `ee_direction_3d` | random-init | `0.537 ± 0.009` | `0.559 ± 0.014` | `0.570 ± 0.010` | `18.7 ± 2.1` | `0.566 ± 0.012` |
| `ee_speed` | pretrained | `0.707 ± 0.027` | `0.930 ± 0.003` | `0.934 ± 0.003` | `11.0 ± 1.6` | `0.917 ± 0.002` |
| `ee_speed` | random-init | `0.582 ± 0.019` | `0.610 ± 0.017` | `0.631 ± 0.013` | `13.0 ± 5.7` | `0.614 ± 0.026` |

### Quantitative deltas

- `ee_direction_3d`:
  - peak `0.816 -> 0.570` (`-0.247`, `-30.2%`)
  - peak layer `11.7 -> 18.7` (`+7.0` layers)
- `ee_speed`:
  - peak `0.934 -> 0.631` (`-0.302`, `-32.4%`)
  - peak layer `11.0 -> 13.0` (`+2.0` layers, high seed variance)

### Why this is a clean null

The random-init probe still fits the training set strongly:

- `ee_direction_3d`: train `L0 = 0.991 ± 0.003`, train peak `= 0.994 ± 0.001`
- `ee_speed`: train `L0 = 0.930 ± 0.036`, train peak `= 0.990 ± 0.005`

The null therefore fails on **validation structure**, not on probe capacity.
The same trainable linear readout can fit the training data on random features,
but it does not recover the pretrained validation regime.

### Verdict

This round rejects the architecture-only explanation for the strongest Push
kinematic results. The V-JEPA 2 Large architecture with random weights still
contains some decodable signal, but it does **not** produce the pretrained
high-R², mid-depth direction and speed regime. The strongest manipulation PEZ
signal is therefore a **learned property of the pretrained representation**,
not a generic consequence of probe capacity or backbone geometry.
