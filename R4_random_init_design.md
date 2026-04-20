# Round 4 Design: Random-Init Null Baseline for Push

## Scope

Round 4 tests whether the current Push PEZ-like curves depend on learned
V-JEPA 2 structure, or whether similar decoding appears even with the same
architecture at random initialization.

Task:

- `push`

Targets:

- `ee_direction_3d`
- `ee_speed`

Model family:

- V-JEPA 2 Large architecture

Null condition:

- identical architecture
- identical extraction recipe
- **random weights**

## Main claim

If the positive Push curves require learned V-JEPA 2 structure, then a random
init backbone should fail to reproduce the same layer-wise behavior.

## Killer evidence chain

1. **Direction should collapse under random init**
   - especially the PEZ-like middle-layer peak
2. **Speed should also weaken materially**
   - or at minimum lose the strong middle-depth rise seen in the learned model
3. **The gap should hold across probe seeds**
   - to rule out optimizer noise

## Counter-evidence scenarios

1. Random-init curves remain close to the learned backbone
   - interpretation: the current result may be architecture bias, not learned
     world modeling
2. Speed survives but direction collapses
   - interpretation: low-level motion magnitude is architecture-accessible,
     but directional structure requires learning
3. Both survive strongly
   - interpretation: the current manipulation story weakens substantially

## Reviewer attack to answer

Reviewer attack:

> "Maybe your token-patch probe is just so strong that any high-capacity
> transformer, even untrained, would yield these curves."

Round-4 answer if successful:

- the same architecture with random weights does **not** produce the same
  direction/speed curves
- therefore the learned V-JEPA 2 representation matters

## Fixed recipe

Keep the representation recipe identical to the successful learned run:

- `resid_post`
- `temporal_last_patch`
- `resize`
- ViT-Large, `24` layers

Probe recipe:

- `trainable` solver
- 20 HP sweep
- `GroupKFold` by `episode_id`
- `zscore`
- probe seeds:
  - `42`
  - `123`
  - `2024`

## Architecture seed

Use a fixed model seed for the random-init backbone:

- `model_seed = 0`

Reason:

- this is the cleanest first null
- if needed later, model-seed variance can be added as a stronger follow-up

## Storage plan

Round 4 needs a new Push token-patch cache.

To make room:

- delete the now-committed F5 intermediate raw cache:
  - `/mnt/md1/solee/features/physprobe_vitl_tokenpatch_shuffled_seed123/push`

New output root:

- `/mnt/md1/solee/features/physprobe_vitl_tokenpatch_randominit_seed0/push`

## Success / failure criteria

### Strong support for learned-representation claim

- `ee_direction_3d` random-init peak `R^2 < 0.3`
- and/or no middle-layer PEZ-like structure

### Mixed support

- direction weakens substantially but remains above `0.3`
- speed weakens only mildly

### Weak support

- random-init results stay close to learned results for both targets

## Outputs

- `artifacts/results/probe_push_ee_direction_3d_large_token_patch_r4_randinit_seed{probe_seed}.csv`
- `artifacts/results/probe_push_ee_speed_large_token_patch_r4_randinit_seed{probe_seed}.csv`
- `artifacts/results/random_init_verdict.md`

## Why Round 4 now

After Round 2 CKA, the main unresolved question is no longer task alignment.

It is:

> are the current Push curves evidence for learned world-model structure, or
> can they arise from the architecture alone?

That makes random-init the highest-value next test.
