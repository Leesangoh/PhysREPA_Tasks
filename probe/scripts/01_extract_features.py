#!/usr/bin/env python3
"""Phase 4: V-JEPA 2 ViT-L feature extraction with caching of Variants A and B.

For each (task, episode):
  - Decode MP4 (camera image_0 by default), preprocess to [3, T, 256, 256] fp32.
  - For windows of size 16 stride 1 (last frame t in {15..T-1}), forward in batches.
  - From each layer's [B, 2048, 1024] activation:
      Variant A = mean over all 2048 tokens → [B, 1024]
      Variant B = spatial mean over 256 tokens → [B, 8, 1024], flattened to [B, 8192]
  - Save cache/<task>/variant_A/episode_<id>.npz and variant_B/episode_<id>.npz.

Multi-GPU is achieved by sharding episodes via --gpu and --shards/--shard-id.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.dataset import decode_mp4_rgb, parquet_episode_ids, video_path
from utils.io import (
    cache_paths,
    list_cached_episodes,
    load_common,
    load_tasks,
    progress,
    save_episode_features,
    write_manifest,
)
from utils.vjepa_loader import load_vjepa2_vit_l, preprocess_frames


WINDOW = 16
T_TOK, N_SP, D = 8, 256, 1024  # ViT-L


def windows_for_clip(
    x_full: torch.Tensor,
    batch: int,
    *,
    shuffle_frames: bool = False,
    rng: np.random.Generator | None = None,
):
    """Yield batched window stacks [B, 3, 16, 256, 256] from a [3, T, 256, 256] clip.

    When ``shuffle_frames`` is True, each emitted window has its 16-frame
    temporal axis permuted by a *fresh* random permutation (unique per window).
    The permutation is drawn from ``rng`` so that the shuffle is deterministic
    given the rng seed. The window's t_last index is unchanged — the same
    `(episode_id, t_last)` keys are produced as the unshuffled run, so all
    downstream alignment with target arrays remains valid.
    """
    T = x_full.shape[1]
    if T < WINDOW:
        return
    n_win = T - WINDOW + 1
    for s in range(0, n_win, batch):
        e = min(s + batch, n_win)
        clips_list = []
        for i in range(s, e):
            clip = x_full[:, i:i + WINDOW]            # [3, 16, 256, 256]
            if shuffle_frames:
                assert rng is not None, "rng required when shuffle_frames=True"
                perm = rng.permutation(WINDOW)
                clip = clip[:, perm, :, :]
            clips_list.append(clip)
        yield s, torch.stack(clips_list)


def extract_episode(
    model,
    task: str,
    episode_id: int,
    batch: int,
    device: str,
    *,
    shuffle_frames: bool = False,
    shuffle_seed: int = 42,
) -> dict[str, np.ndarray]:
    """Returns {feats_A [N_win, 24, 1024] fp16, feats_B [N_win, 24, 8192] fp16, t_last [N_win]}.

    ``shuffle_frames=True`` permutes the 16-frame stack of every input window
    with a fresh random permutation (per-window). RNG is seeded as
    ``shuffle_seed XOR episode_id`` so the same shuffle reproduces across
    re-extraction runs.
    """
    arr = decode_mp4_rgb(video_path(task, episode_id))
    T = arr.shape[0]
    n_win = T - WINDOW + 1
    if n_win <= 0:
        return {}
    frames = torch.from_numpy(arr.copy()).to(device)
    x_full = preprocess_frames(frames).float()       # [3, T, 256, 256] fp32

    feats_A = np.zeros((n_win, 24, D), dtype=np.float16)
    feats_B = np.zeros((n_win, 24, T_TOK * D), dtype=np.float16)
    t_last = np.arange(WINDOW - 1, T, dtype=np.int32)

    rng = np.random.default_rng(shuffle_seed ^ int(episode_id)) if shuffle_frames else None

    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
        for s, clips in windows_for_clip(x_full, batch, shuffle_frames=shuffle_frames, rng=rng):
            outs = model(clips)                       # list of 24 × [B, 2048, 1024] fp16
            B = clips.shape[0]
            for layer_idx, act in enumerate(outs):
                # Cast to fp32 BEFORE pooling so that A and B agree under the
                # pooling identity (cache only, fp16 storage afterwards).
                a = act.float().reshape(B, T_TOK, N_SP, D)
                varA = a.mean(dim=(1, 2))             # [B, 1024]
                varB = a.mean(dim=2).reshape(B, T_TOK * D)  # [B, 8192]
                feats_A[s:s + B, layer_idx] = varA.to(torch.float16).cpu().numpy()
                feats_B[s:s + B, layer_idx] = varB.to(torch.float16).cpu().numpy()
    return {"feats_A": feats_A, "feats_B": feats_B, "t_last": t_last}


def run_task(
    task: str,
    *,
    gpu: int,
    shards: int,
    shard_id: int,
    batch: int,
    max_episodes: int | None,
    shuffle_frames: bool = False,
    shuffle_seed: int = 42,
):
    device = f"cuda:{gpu}"
    torch.cuda.set_device(gpu)
    common = load_common()
    cfg = load_tasks()[task]

    progress(f"[extract] {task} loading model on {device} (shuffle_frames={shuffle_frames})")
    t0 = time.time()
    model, _ = load_vjepa2_vit_l(device)
    progress(f"[extract] {task} model loaded {time.time()-t0:.1f}s")

    eps_all = parquet_episode_ids(task)
    eps = [e for i, e in enumerate(eps_all) if i % shards == shard_id]
    if max_episodes is not None:
        eps = eps[:max_episodes]

    # Variant naming: "A" / "B" for unshuffled cache; "A_shuffled" for shuffle cache
    # (Variant B is not saved under shuffle to cap storage, per Plan Phase 3 spec).
    var_a = "A_shuffled" if shuffle_frames else "A"
    var_b = "B"   # only used when shuffle_frames is False
    cached_A = set(list_cached_episodes(task, var_a))
    if shuffle_frames:
        # Only need A to be present; B is not extracted under shuffle.
        todo = [e for e in eps if e not in cached_A]
    else:
        cached_B = set(list_cached_episodes(task, var_b))
        todo = [e for e in eps if (e not in cached_A) or (e not in cached_B)]

    progress(
        f"[extract] {task} shard {shard_id+1}/{shards} batch={batch} "
        f"variant={var_a}{'' if shuffle_frames else '+'+var_b}: "
        f"total {len(eps)} todo {len(todo)}"
    )

    n_win_total = 0
    t_start = time.time()
    for i, ep in enumerate(todo):
        d = extract_episode(
            model, task, ep, batch, device,
            shuffle_frames=shuffle_frames,
            shuffle_seed=shuffle_seed,
        )
        if not d:
            progress(f"[extract] {task} ep {ep}: skipped (T<{WINDOW})")
            continue
        save_episode_features(task, var_a, ep, d["feats_A"], d["t_last"])
        if not shuffle_frames:
            save_episode_features(task, var_b, ep, d["feats_B"], d["t_last"])
        n_win_total += d["t_last"].size

        if (i + 1) % 25 == 0 or (i + 1) == len(todo):
            dt = time.time() - t_start
            wps = n_win_total / max(dt, 1e-6)
            eta = (len(todo) - i - 1) * dt / max(i + 1, 1)
            progress(f"[extract] {task} sh{shard_id} {i+1}/{len(todo)} eps "
                     f"{n_win_total} win {wps:.1f} win/s ETA {eta/60:.1f}min")

    progress(f"[extract] {task} sh{shard_id} DONE: {len(todo)} eps {n_win_total} win in {(time.time()-t_start)/60:.1f}min")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument(
        "--shuffle-frames",
        action="store_true",
        help="Per-window random permutation of the 16-frame stack BEFORE the V-JEPA "
             "forward pass. Saves Variant A only to cache/<task>/variant_A_shuffled/. "
             "(Phase 3 / F5 frame-shuffle ablation.)",
    )
    p.add_argument(
        "--shuffle-seed",
        type=int,
        default=42,
        help="Seed for shuffle RNG. Per-episode rng = shuffle_seed XOR episode_id.",
    )
    args = p.parse_args()
    run_task(
        args.task,
        gpu=args.gpu,
        shards=args.shards,
        shard_id=args.shard_id,
        batch=args.batch,
        max_episodes=args.max_episodes,
        shuffle_frames=args.shuffle_frames,
        shuffle_seed=args.shuffle_seed,
    )


if __name__ == "__main__":
    main()
