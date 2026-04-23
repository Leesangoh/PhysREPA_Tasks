# Scope Expansion Verdict: PegInsert and NutThread

## Setup

- Backbone: `V-JEPA 2 Large`
- Cache type: existing `mean` feature windows from
  `/mnt/md1/solee/features/physprobe_vitl/{peg_insert,nut_thread}`
- Discovery seed: `42`
- Tightening seeds for the strongest constrained-contact target: `123`, `2024`
- Targets:
  - `peg_insert`: `ee_direction_3d`, `ee_speed`, `insertion_depth`,
    `peg_hole_lateral_error`
  - `nut_thread`: `ee_direction_3d`, `ee_speed`, `axial_progress`,
    `nut_bolt_relative_angle`

## Main findings

### PegInsert

- `ee_direction_3d` is the only clear PEZ-style scope-expansion positive:
  - seed `42`: peak `0.5568 @ L12/24`
  - seed `123`: peak `0.5543 @ L20/24`
  - seed `2024`: peak `0.5550 @ L12/24`
- The apparent `L20` peak on seed `123` is not a genuine late-only shift.
  - On that seed, `L12=0.5539` and `L20=0.5543`, a gap of only `+0.00038`
  - The curve therefore forms a broad high plateau from roughly `L12` through
    `L20`, not a monotonic last-layer refinement pattern
- Across all three seeds:
  - peak `R^2 = 0.5554 ± 0.0010`
  - `L12` value `= 0.5552 ± 0.0012`
  - two of three seeds are maximized exactly at `L12`
- Other PegInsert targets are strong but late:
  - `ee_speed`: `0.6853 @ L17`
  - `insertion_depth`: `0.9165 @ L20`
  - `peg_hole_lateral_error`: `0.6184 @ L17`

Interpretation:
- PegInsert extends the manipulation PEZ story beyond Push/Strike into one
  constrained insertion regime.
- The most defensible wording is not “all insertion tasks show PEZ,” but
  “PegInsert exhibits a stable mid-to-upper-mid direction plateau consistent
  with PEZ-like transfer.”

### NutThread

- `ee_direction_3d` is weak and late:
  - peak `0.1622 @ L22/24`
- Task-specific progress variables are decodable, but only near the output:
  - `ee_speed`: `0.5390 @ L21`
  - `axial_progress`: `0.6706 @ L21`
  - `nut_bolt_relative_angle`: `0.4602 @ L22`

Interpretation:
- NutThread does not currently support the PEZ generalization story.
- Under the current cache and target formulation, its accessible signals are
  late-peaking and weaker than PegInsert.

## Bottom line

- Scope expansion is now more nuanced and stronger than before:
  - `Push`, `Strike`, and now `PegInsert` show PEZ-consistent intermediate or
    broad mid-depth direction accessibility
  - `NutThread` does not
- This weakens the simplest “PEZ is universal across manipulation” story, but
  strengthens the more accurate claim:
  - PEZ-like transfer generalizes beyond free sliding and striking into at
    least one constrained insertion task, while remaining task-dependent.
