# ORAL STRATEGY

## Executive assessment

With the current evidence alone, this project is **not yet at NeurIPS Oral level**.

It is already a strong paper:

- manipulation PEZ-like direction emergence exists
- `3D` parameterization matters materially
- temporal corruption shifts the emergence layer
- cross-task CKA falsifies the naive “shared PEZ layer” story and replaces it
  with a stronger specialization account

That is enough for a serious conference paper.

But **oral** requires more than “a strong set of probing results.”

It requires a result that a reviewer would describe as:

1. **mechanistically decisive**
2. **broadly general**
3. **unexpected or paradigm-shifting**
4. ideally **useful beyond pure analysis**

So the honest answer is:

> **Yes, oral is possible in principle, but only if this project expands from a
> single-model probing study into a full evidence chain that establishes a new
> phenomenon, a mechanism, and a downstream consequence.**

This means the current question is no longer “what can be added quickly?”

It is:

> **what is the strongest scientific claim this project could legitimately make,
> and what evidence would force reviewers to believe it?**

## The strongest possible oral-grade claim

The strongest version of this work is **not**:

- “PEZ transfers to manipulation”
- “V-JEPA 2 also has PEZ-like curves”

Those are interesting, but too descriptive.

The strongest oral-grade claim is:

> **Video world models develop a temporally grounded, learned, task-specializing
> kinematic computation layer.**

More concretely:

1. this layer is **not present in random networks**
2. it is **not a static visual correlate**
3. it is **not unique to one task**
4. it is **not unique to one model size**
5. it is **not unique to one backbone family**
6. and it is **functionally useful** for control under physics variation

If all six are demonstrated, then the project becomes much closer to oral-tier.

## What is uniquely novel here?

There are three genuinely distinctive contributions already visible in the data.

These should be elevated, not buried.

### 1. 3D parameterization changes the scientific conclusion

This is not a cosmetic implementation detail.

It is a methodological result:

- `2D direction` looked weak or inconsistent in manipulation
- `3D direction` rescued the phenomenon, especially for `strike / object_direction`

That means:

> **PEZ-style probing in embodied domains is highly sensitive to target
> geometry, and incorrect parameterization can erase real structure.**

This is broadly useful and portable.

It says something not just about V-JEPA 2, but about how interpretability
analyses should be designed in robotics.

### 2. Peak-layer shift under temporal corruption is a new signal

Most papers only look at peak score collapse.

Here, the more revealing effect under frame shuffle was:

- the representation still decodes something
- but the **emergence layer shifts dramatically later**

That is a qualitatively different insight:

> temporal corruption can preserve decodability while forcing the hierarchy to
> solve the problem later.

This suggests a new tool for studying temporal representation:

- not only “does the signal survive?”
- but “where in the hierarchy is the signal forced to re-emerge?”

That is a strong conceptual contribution if shown robustly across models/tasks.

### 3. PEZ is task-specializing, not task-general

The CKA result is scientifically valuable.

The naive story would have been:

- “there is a universal physics layer shared by all tasks”

But the actual result is more interesting:

- early layers are generic
- PEZ-like layers are where tasks become more separable
- late layers flatten rather than re-converge

That turns the work from “shared hidden layer” into a new story about
**specialization**.

This is richer and more believable.

## What is still missing for oral?

There are four missing components.

### Missing component 1: learned-vs-unlearned causality

Current evidence is still vulnerable to:

> “maybe the architecture is enough”

Round 4 random-init is the first answer.

But for oral-level confidence, this likely needs:

- multiple random seeds
- maybe multiple architectures with matched depth/width if feasible

### Missing component 2: cross-model generality

Right now the phenomenon is tied to V-JEPA 2.

Oral reviewers will ask:

> “Is this a V-JEPA quirk, or a general property of video SSL world models?”

To answer that, this work should compare:

- **V-JEPA 2**
- **VideoMAE**
- **Hiera** or another video transformer
- ideally **DINOv2** as an image-only control

This is crucial because the most interesting answer may be:

- temporal PEZ-like emergence is strong in video-pretrained models
- weak or absent in image-only models

That would be a genuinely new scientific statement.

### Missing component 3: scale law

Within V-JEPA 2 alone, the project should establish whether PEZ-like emergence
obeys a scaling regularity:

- Large
- Giant
- Huge

If the emergence band shifts predictably with scale, that is a memorable oral
figure.

Without scale, the story still looks like a one-model observational study.

### Missing component 4: downstream functional consequence

This is the biggest difference between a strong analysis paper and an oral-level
paper.

If PEZ-like layers are truly special, then they should be **more useful** than
other layers for something meaningful.

The strongest downstream candidate is:

> **PEZ-aware policy learning or policy representation selection under physics
> shift.**

For example:

- behavior cloning with frozen backbone features from each layer
- compare success/generalization under unseen mass/friction/restitution
- test whether PEZ-like layers outperform shallow and very late layers

If that holds, then the paper is no longer just interpretability.

It becomes:

> “we discovered a representational regime that also improves control.”

That is oral-level material.

## Full-scope evidence chain for oral

If there were no time constraints, the evidence chain I would build is:

### Chain A: Existence

Show PEZ-like kinematic emergence on **all 6 PhysProbe tasks**:

- Push
- Strike
- PegInsert
- NutThread
- Drawer
- Reach

Targets:

- `ee_direction_3d`
- `ee_speed`
- `ee_accel_magnitude`
- task-appropriate object direction where meaningful

Negative controls:

- `fake_mod5`
- random target permutations

Goal:

- establish that the phenomenon is real and broad, not cherry-picked

### Chain B: Geometry

Show that target parameterization matters:

- `2D angle`
- `2D sin/cos`
- `3D direction`
- raw velocity vector if useful

Goal:

- establish that **3D is the correct scientific probe target** for manipulation
- elevate this from “implementation detail” to “methodological discovery”

### Chain C: Temporal mechanism

Use multiple temporal corruptions:

1. frame shuffle
2. reverse order
3. single-frame control
4. frame-repeat / hold-last control
5. maybe temporal subsampling / coarse time warp

Measure:

- `delta_peak`
- `delta_Lpez`
- emergence-layer shift

Goal:

- prove that PEZ-like emergence is tied to temporal dynamics, not just visual
  appearance

### Chain D: Learned representation

Compare learned V-JEPA 2 against:

- random-init V-JEPA 2
- maybe multiple random seeds

Goal:

- prove the phenomenon is learned, not architectural

### Chain E: Scale law

Run:

- V-JEPA 2 Large
- V-JEPA 2 Giant
- V-JEPA 2 Huge

Measure:

- emergence layer
- peak layer
- late decline or plateau

Goal:

- reveal a systematic scaling law

### Chain F: Cross-model taxonomy

Run the same pipeline on:

- V-JEPA 2
- VideoMAE
- Hiera
- DINOv2

The desired taxonomy would be:

- video world models: strong temporal PEZ-like effect
- image-only models: weaker or missing temporal effect

Goal:

- turn this from a model-specific observation into a representational taxonomy

### Chain G: Task specialization

Keep the CKA result, but deepen it:

- same-layer CKA across all tasks
- inter-layer transfer matrix
- probe transfer across tasks

Goal:

- show that PEZ-like layers are where tasks specialize, not globally align

This is the “unexpected” result that makes the story deeper.

### Chain H: Downstream consequence

Use frozen features from every layer for:

- behavior cloning
- inverse dynamics
- forward prediction
- physics-parameter inference under distribution shift

Most compelling:

- layer-wise policy training under held-out physics parameters

Goal:

- show that PEZ-like layers are not just interpretable, but **useful**

This is the strongest possible oral booster.

## Ranked list of oral-grade additions

If all of the above is too much, here is the ranked order of what matters most.

### Tier 1: essential

1. **Random-init baseline**
2. **Scale law (`L/G/H`)**
3. **At least one stronger temporal null than shuffle**
4. **All 6 tasks**

If these four are done well, the paper becomes much more serious.

### Tier 2: likely oral-making

5. **Cross-model comparison (`V-JEPA 2` vs `VideoMAE` vs `Hiera` vs `DINOv2`)**
6. **Downstream policy relevance**

If either of these works strongly, oral becomes much more plausible.

### Tier 3: excellent amplifiers

7. **Inter-layer transfer matrix**
8. **Task-pair probe transfer**
9. **More principled theory / scaling analysis of motion magnitude vs emergence**

These enrich the paper, but are less likely than Tier 2 to be the decisive
factor.

## What would I do first, in full scope?

If the goal is to maximize oral probability, I would sequence the project as:

### Phase O1: lock the causal nulls

1. Finish random-init baseline
2. Add single-frame control
3. Add one more temporal corruption beyond shuffle

This determines whether the temporal claim is genuinely strong or only mixed.

### Phase O2: establish structure

4. Run `L/G/H` scale law on the strongest targets
5. Complete all 6 tasks with the validated target parameterization (`3D`)

This creates the core empirical figure set.

### Phase O3: establish generality

6. Run cross-model comparison
   - V-JEPA 2
   - VideoMAE
   - Hiera
   - DINOv2

This is where the paper becomes field-level rather than one-model-specific.

### Phase O4: establish utility

7. Do downstream policy / control experiments

This is the single highest upside addition.

If PEZ-like layers improve policy generalization under physics shift, the paper
becomes much more than an interpretability paper.

## Honest conclusion

If the project stays at:

- one pretraining family
- one main analysis pipeline
- no downstream consequence

then I would still expect:

- strong paper
- maybe spotlight depending on reviewer taste
- **not reliably oral**

If the project adds:

1. random-init failure
2. scale law
3. cross-model taxonomy
4. downstream control benefit

then it has a **real oral path**.

The most important shift is conceptual:

> the paper should not be sold as “PEZ exists in manipulation too.”

It should be sold as:

> **video world models develop learned, temporally grounded, task-specializing
> kinematic computation layers, and these layers can be identified,
> stress-tested, and exploited.**

That is the version of the story that can plausibly reach oral quality.
