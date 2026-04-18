# Phase 2 Blocker: Push Token-Patch Cache

## Status

Phase 2 was blocked before the full Push token-level run.

The requested cache layout was:

- capture: `resid_post`
- transform: `resize`
- pooling: `temporal_last_patch`
- output layout: per-episode safetensors with keys
  - `window_starts`
  - `layer_{L}_window_{W}` of shape `(n_patches, D)`

This layout was implemented and validated on a real Push episode, but it is not
feasible to run on the full 1500-episode Push split with the currently
available disk space.

## Dry-run validation

One-episode dry-run command:

```bash
/isaac-sim/python.sh extract_token_features.py \
  --task push \
  --model large \
  --output-root artifacts/tokenpatch_dryrun \
  --device cuda:0 \
  --batch-size 2 \
  --episode-limit 1 \
  --overwrite
```

Observed output:

- `window_starts.shape = (58,)`
- `layer_0_window_0.shape = (256, 1024)` with `float16`
- `layer_23_window_57.shape = (256, 1024)`
- single-episode file size: `729,937,416 bytes` (`696.1 MiB`)

This confirms that the extractor is producing the intended PEZ-style
`temporal_last_patch` token tensors.

## Storage math

Empirical extrapolation from the real dry-run file:

- per-episode cache size: `696.1 MiB`
- Push episodes: `1500`
- required storage: `696.1 MiB * 1500 = 0.996 TiB`

Measured free space:

- `/home/solee` / `/mnt/md1` visible free space during the run: `166G`

Therefore:

- required: `~1.0 TiB`
- available: `166G`
- deficit: `~830G`

## Decision

Following the Phase 2 instruction, the experiment was **not** continued on a
subset because that would be a false negative / false positive risk for the
Push verdict.

Phase 2 is therefore blocked on cache format / storage, not on model loading or
token extraction correctness.

## Next technically valid workaround

The current probe design averages windows within each episode before fitting the
linear probe. Because of that, a mathematically equivalent cache format is:

- store per-layer episode-mean token patches only
- shape per layer: `(256, 1024)`
- no per-window tensors on disk

This would reduce storage from `~1.0 TiB` to roughly `~18-19 GiB` for Push and
preserve the exact episode-level features used by the current probe.

This workaround was **not** executed in this step because the Phase 2 request
explicitly specified per-window token safetensors.
