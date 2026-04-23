# Phase 3 Blocker: Event-Aligned Force Probe

## Status

`BLOCKED`

## Intended experiment

The planned Phase 3 follow-up after the Phase 2 partial PEZ success was:

- window-level probing
- targets:
  - `contact_flag` (binary classification)
  - `contact_force_magnitude` (scalar regression)
- grouping:
  - `GroupKFold by episode_id`
- feature recipe:
  - `resid_post`
  - `temporal_last_patch`
  - `trainable 20-HP sweep`

## Why it is blocked

The required contact labels appear to be absent in the currently available public PhysProbe release.

### Push full audit

- channel: `physics_gt.contact_flag`
- episodes scanned: `1500 / 1500`
- non-zero episodes: `0`
- maximum per-episode mean: `0.0`

### Push sampled contact channels

All sampled values were identically zero for:

- `physics_gt.contact_flag`
- `physics_gt.contact_force`
- `physics_gt.contact_point`
- `physics_gt.contact_finger_l_object_flag`
- `physics_gt.contact_finger_l_object_force`
- `physics_gt.contact_object_surface_flag`
- `physics_gt.contact_object_surface_force`

### Cross-task sanity sample

Sampled episodes from the following tasks also showed all-zero contact channels:

- `strike`
- `peg_insert`
- `nut_thread`
- `drawer`

## Interpretation

This is a dataset-label blocker, not a model/probe blocker.

Launching a force/contact probe on these labels would produce a meaningless result:

- binary target would have zero variance
- scalar force magnitude would be identically zero
- any reported curve would reflect label pathology, not representation quality

## Recommended next step

Use one of these alternatives:

1. Obtain a PhysProbe release with populated contact labels.
2. Define an event-aligned target from non-zero kinematic/object-state channels instead of the current contact fields.
3. Continue scaling the successful Phase 2 recipe to other tasks/models before returning to event probes.
