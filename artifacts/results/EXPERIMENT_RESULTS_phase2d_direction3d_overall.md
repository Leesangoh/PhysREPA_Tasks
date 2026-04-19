# Phase 2d Overall Verdict: 2D vs 3D Direction in Manipulation

Date: 2026-04-19

## Question

The PEZ paper uses a planar direction target because its Kubric setup is overhead 2D motion.
PhysProbe manipulation trajectories are genuinely 3D. This phase asks:

- when should manipulation direction be modeled as 2D XY-angle (`sin/cos`)?
- when should it be modeled as full 3D unit direction (`v / ||v||`)?

This document consolidates the final comparison across all three tasks run with the PEZ-aligned token-patch recipe:

- `push`
- `reach`
- `strike`

Shared recipe:

- model: `V-JEPA 2 Large`
- capture: `resid_post`
- pooling: `temporal_last_patch`
- feature type: `token_patch`
- probe: `trainable 20-HP sweep`
- CV: `5-fold GroupKFold by episode_id`
- norm: `zscore`

## Consolidated Table

| Task | Target | L0 | L8 | Peak R2 | Peak layer | Last | Classification |
|---|---|---:|---:|---:|---:|---:|---|
| Push | `ee_direction_sincos` | 0.555 | 0.789 | 0.807 | 13 | 0.797 | `PEZ-like` |
| Push | `ee_direction_3d` | 0.652 | 0.806 | 0.817 | 11 | 0.813 | `PEZ-like` |
| Push | `object_direction_sincos` | -0.220 | 0.048 | 0.084 | 23 | 0.084 | `fail` |
| Push | `object_direction_3d` | -0.160 | 0.097 | 0.136 | 23 | 0.136 | `never-linear` |
| Reach | `ee_direction_sincos` | 0.302 | 0.353 | 0.396 | 20 | 0.395 | `intermediate` |
| Reach | `ee_direction_3d` | 0.435 | 0.536 | 0.553 | 11 | 0.502 | `PEZ-like` |
| Strike | `ee_direction_sincos` | 0.697 | 0.871 | 0.885 | 11 | 0.877 | `PEZ-like` |
| Strike | `ee_direction_3d` | 0.623 | 0.818 | 0.849 | 22 | 0.847 | `intermediate` |
| Strike | `object_direction_sincos` | -0.220 | 0.048 | 0.084 | 23 | 0.084 | `fail` |
| Strike | `object_direction_3d` | 0.521 | 0.774 | 0.813 | 12 | 0.812 | `PEZ-like` |

## Task-by-Task Interpretation

### Push

#### End-effector direction

`ee_direction_sincos` and `ee_direction_3d` are both positive.

- 2D:
  - `peak=0.807 @ L13`
- 3D:
  - `peak=0.817 @ L11`

Interpretation:

- Push arm motion is rich enough that both parameterizations recover a PEZ-like curve
- 3D is slightly stronger in absolute R2 and slightly earlier in peak
- this supports the user’s critique: the final manipulation result should not rely only on a planar target

#### Object direction

- 2D: `peak=0.084 @ L23`
- 3D: `peak=0.136 @ L23`

Interpretation:

- 3D helps slightly, but Push object direction still fails
- likely reason: the pushed object has low-speed / intermittent motion and too much near-zero velocity mass

### Reach

#### End-effector direction

- 2D: `peak=0.396 @ L20`
- 3D: `peak=0.553 @ L11`

Interpretation:

- this is a decisive upgrade
- 2D made Reach look non-PEZ-like
- 3D turns it into a legitimate PEZ-like direction result

Conclusion:

- Reach is the strongest evidence that 2D was the wrong scientific target for manipulation direction

### Strike

#### End-effector direction

- 2D: `peak=0.885 @ L11`
- 3D: `peak=0.849 @ L22`

Interpretation:

- 2D remains better for clean PEZ-shape on the arm side in Strike
- 3D still yields very high decodability, but the peak shifts too late

#### Object direction

- 2D: `peak=0.084 @ L23`
- 3D: `peak=0.813 @ L12`

Interpretation:

- this is the biggest reversal in the entire project
- planar XY target essentially erased the signal
- 3D reveals a strong, mid-depth, PEZ-like object-direction representation

Conclusion:

- for object motion in contact-heavy manipulation, 3D is not optional; it is the correct target

## Global Takeaways

### 1. 2D direction is not a safe default for manipulation

It worked as a direct Kubric analog, but it can materially understate manipulation representations:

- `reach / ee_direction`: undercalled by 2D
- `strike / object_direction`: almost completely hidden by 2D

### 2. 3D direction is the better scientific default

Across tasks, 3D direction:

- improves Reach from weak to PEZ-like
- preserves Push ee-side positivity
- rescues Strike object direction dramatically

### 3. 2D is still useful as a paper-aligned auxiliary target

There is still one place where 2D is arguably preferable:

- `strike / ee_direction_sincos` gives the cleanest mid-depth PEZ shape

So the most defensible position is:

- **3D direction = default manipulation target**
- **2D direction = auxiliary Kubric-analog diagnostic**

## Final Verdict

### If the question is “does manipulation show PEZ-like direction using a scientifically correct 3D target?”

Answer: **yes**

Evidence:

- `push / ee_direction_3d`: `peak=0.817 @ L11`
- `reach / ee_direction_3d`: `peak=0.553 @ L11`
- `strike / object_direction_3d`: `peak=0.813 @ L12`

### If the question is “is the paper-style 2D direction target enough for manipulation?”

Answer: **no**

Evidence:

- it misses Reach’s mid-depth arm-direction signal
- it almost completely misses Strike’s object-direction signal

## Recommended Reporting Language

The strongest honest summary is:

> The PEZ methodology transfers to manipulation data, but direction must be treated as a 3D target rather than a planar Kubric-style target. Using a 3D direction parameterization strengthens Reach and dramatically rescues Strike object-direction decoding.
