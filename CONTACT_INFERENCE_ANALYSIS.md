# Contact Inference Analysis

## Goal

Re-evaluate the previous Phase 3 blocker.

The earlier blocker was:

- exported `contact_flag`, `contact_force`, and related `contact_*` fields were
  zero-filled in the public PhysProbe parquet exports
- therefore a direct event-aligned contact probe looked impossible

The key correction is:

> zero-filled `contact_*` GT does **not** imply there is no recoverable
> interaction-event signal in the data.

In particular, `strike` already shows large object-acceleration spikes despite
all-zero `contact_flag/contact_force`.

## What was checked

### 1. Column availability

A direct parquet audit shows:

- `push`, `strike`, `peg_insert`, `nut_thread`, `drawer` all contain
  `physics_gt.contact_*` columns
- `reach` does not, which is expected

### 2. Direct strike sanity check

For `strike / episode_000000`:

- `object_speed` ranges roughly `0.027 -> 0.281`
- `object_acceleration_magnitude` peaks at **`13.99`**
- top-5 acceleration peaks:
  - `13.99`
  - `8.05`
  - `6.24`
  - `3.44`
  - `2.91`
- meanwhile:
  - `contact_flag` nonzero count = `0`
  - `contact_force` nonzero count = `0`

This is already enough to show:

> contact-like impulses are present in the kinematics even when the explicit
> contact channels are unusable.

### 3. Multi-episode audit (sampled)

A quick audit over the first `50` parquet episodes per manipulation task showed:

- all exported `contact_*` fields had nonzero fraction `0.0`
- therefore the public contact GT is unusable as a supervision target

But kinematic proxies were clearly task-dependent.

## Evidence by task

## Push

### Strongest usable signal

**Direct object acceleration magnitude** from `physics_gt.object_acceleration`

Sampled `50`-episode summary:

- acceleration max mean: `11.72`
- acceleration max median: `10.91`
- baseline median acceleration mean: `0.052`

This is an enormous spike-to-baseline ratio.

### Supporting signals

- `ee_to_object_distance` derivative
  - useful as an approach/compression cue
  - but weaker than acceleration spikes
- `object_velocity_direction` change
  - informative but noisier than raw acceleration peaks

### Recommended surrogate labels

- `contact_happening_t`
  - `1` if `object_acceleration_magnitude_t` exceeds an episode-adaptive robust
    threshold
  - recommended threshold:
    - `median + 5 * MAD`
    - or top `2%` within the episode
- `contact_force_proxy_t`
  - `object_mass * object_acceleration_magnitude_t`

### Assessment

**High feasibility**

Push does not have the cleanest object event signal, but it is clearly usable.

## Strike

### Strongest usable signal

**Direct object acceleration magnitude** from `physics_gt.object_acceleration`

Sampled `50`-episode summary:

- acceleration max mean: `27.74`
- acceleration max median: `25.22`
- baseline median acceleration mean: `0.049`

This is the cleanest contact-impulse proxy in the current dataset.

### Supporting signals

- `object_velocity_direction` jump
  - strong collision-induced redirection/reversal signal
- `ee_to_object_distance` derivative
  - useful as approach context

### Recommended surrogate labels

- `contact_happening_t`
  - `1` if `object_acceleration_magnitude_t` exceeds a robust episode threshold
- `contact_force_proxy_t`
  - `object_mass * object_acceleration_magnitude_t`
- optional richer target:
  - `impact_direction_change_t`
  - based on consecutive object-velocity angle jumps

### Assessment

**Very high feasibility**

If Phase 3 is restarted, `strike` should be the first task.

## PegInsert

### What is missing

There is no direct `peg_acceleration` column in the public export.

So the best options come from:

- finite-difference acceleration from `peg_velocity`
- progress/change signals from:
  - `insertion_depth`
  - `peg_hole_lateral_error`

### Sampled signals

`50`-episode finite-difference summary:

- fd-acceleration max mean: `0.084`
- fd-acceleration max median: `0.069`
- baseline median fd-acceleration mean: `0.042`

This is much weaker than Push/Strike.

However:

- `insertion_depth` delta is available
- `peg_hole_lateral_error` is available

### Recommended surrogate labels

Use a **composite label**, not a single scalar threshold:

- `contact_happening_t = 1` if either:
  - finite-difference peg acceleration is in the top episode percentile
  - or insertion-depth change spikes while lateral error drops

Possible rule:

- `fd_accel_mag_t > episode_p98`
  OR
- `Δinsertion_depth_t > episode_p98` and `Δpeg_hole_lateral_error_t < episode_p02`

### Assessment

**Medium feasibility**

Likely workable, but much noisier than Strike.

## NutThread

### What is missing

There is no direct `nut_acceleration` column in the public export.

Available candidates:

- finite-difference acceleration from `nut_velocity`
- `axial_progress`
- `nut_bolt_relative_angle`

### Sampled signals

`50`-episode finite-difference summary:

- fd-acceleration max mean: `0.073`
- fd-acceleration max median: `0.066`
- baseline median fd-acceleration mean: `0.044`

Again, weaker than Push/Strike.

But this task has a better task-specific interaction descriptor than Push:

- `nut_bolt_relative_angle`
- `axial_progress`

### Recommended surrogate labels

Composite event rule:

- `contact_happening_t = 1` if either:
  - `fd_accel_nut_mag_t > episode_p98`
  - or `|Δnut_bolt_relative_angle_t|` spikes together with
    positive `Δaxial_progress_t`

Potential force proxy:

- `nut_mass * fd_accel_nut_mag_t`

### Assessment

**Medium feasibility**

Better than “blocked,” but still secondary after Strike and Push.

## Drawer

### Important caveat

The current public export appears to have zero-filled handle kinematics:

For sampled `drawer / episode_000000`:

- `handle_position` magnitude = `0`
- `handle_velocity` magnitude = `0`

Across the sampled audit:

- finite-difference handle acceleration = `0`
- position second difference = `0`

So the object-side contact signal is not recoverable in the same way.

### What remains available

- `drawer_joint_pos`
- `drawer_opening_extent`
- `ee_position`

### Recommended surrogate labels

Only a coarse onset-style event is realistic:

- `contact_happening_t = 1` if
  - `|Δdrawer_joint_pos_t|` or
  - `|Δdrawer_opening_extent_t|`
  exceeds an episode threshold

This is not a true impact/contact proxy.

It is closer to:

- “drawer motion onset”

than:

- “contact impulse”

### Assessment

**Low feasibility for true contact probing**

Drawer is currently the weakest task for Phase 3 restart.

## Reach

Reach has no object-contact fields and no contact interpretation target.

It remains useful only as:

- a non-contact negative control

not as a Phase 3 event task.

## Recommended task order for Phase 3 restart

### Priority 1

1. **Strike**
2. **Push**

Reason:

- direct acceleration exists
- spike-to-baseline ratio is very strong
- event labels can be defined cleanly at the frame/window level

### Priority 2

3. **PegInsert**
4. **NutThread**

Reason:

- require finite-difference and composite heuristics
- noisier but still plausible

### Priority 3

5. **Drawer**

Reason:

- no trustworthy object/handle kinematics in the public export
- only coarse motion-onset surrogate is available

## Recommended surrogate labels by task

| Task | Best binary label | Best scalar proxy | Confidence |
|---|---|---|---|
| Push | `object_accel_mag > robust threshold` | `mass * object_accel_mag` | High |
| Strike | `object_accel_mag > robust threshold` | `mass * object_accel_mag` | Very high |
| PegInsert | `fd_peg_accel spike OR insertion-depth event` | `peg_mass * fd_peg_accel_mag` | Medium |
| NutThread | `fd_nut_accel spike OR angle+axial event` | `nut_mass * fd_nut_accel_mag` | Medium |
| Drawer | `drawer_joint/opening onset` | weak / not recommended | Low |

## Phase 3 restart plan

The previous Phase 3 attempt was blocked because it assumed:

- valid `contact_flag`
- valid `contact_force`

That assumption is now obsolete.

### New Phase 3 should be window-level

Use the existing token-patch cache and define labels per 16-frame window.

For each window:

- `contact_happening_window = 1`
  - if **any frame** in the window crosses the event threshold
- `contact_force_proxy_window`
  - `max_t (mass * accel_mag_t)` over the window
  - or `mean_t` over positive frames only

### Recommended first implementation

New script:

- `probe_events.py`

Inputs:

- token-patch cache
- parquet-derived per-frame surrogate labels

Targets:

- binary classification:
  - `contact_happening_window`
- regression:
  - `contact_force_proxy_window`

Grouping:

- `GroupKFold` by `episode_id`

Recommended first task:

- `strike`

Then:

- `push`

## Why this matters scientifically

This could sharpen the paper story substantially.

Right now the manipulation PEZ story is mostly about:

- direction
- speed
- acceleration magnitude

If Phase 3 succeeds with kinematic-derived contact surrogates, the story becomes:

> PEZ-like layers are where **interaction events** become more decodable, not
> just where motion direction becomes decodable.

That is a stronger and more robotics-specific claim.

Potential oral-level value:

- it links temporal PEZ-like structure to **contact-rich decision moments**
- it moves the project closer to mechanistic interaction understanding
- it gives a clearer bridge to downstream control/policy relevance

## Bottom line

The earlier “contact is impossible because contact_flag is zero” conclusion was
too pessimistic.

The updated conclusion is:

1. exported contact GT is unusable
2. **kinematic surrogate contact labels are still feasible**
3. this is strongest for:
   - `strike`
   - then `push`
4. Phase 3 should be restarted as a **window-level event probe** using
   acceleration-derived pseudo-contact targets

That restart is justified by the current data.
