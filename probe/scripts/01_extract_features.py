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


def windows_for_clip(x_full: torch.Tensor, batch: int):
    """Yield batched window stacks [B, 3, 16, 256, 256] from a [3, T, 256, 256] clip."""
    T = x_full.shape[1]
    if T < WINDOW:
        return
    n_win = T - WINDOW + 1
    for s in range(0, n_win, batch):
        e = min(s + batch, n_win)
        yield s, torch.stack([x_full[:, i:i + WINDOW] for i in range(s, e)])


def extract_episode(model, task: str, episode_id: int, batch: int, device: str) -> dict[str, np.ndarray]:
    """Returns {feats_A [N_win, 24, 1024] fp16, feats_B [N_win, 24, 8192] fp16, t_last [N_win]}."""
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

    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
        for s, clips in windows_for_clip(x_full, batch):
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


def run_task(task: str, *, gpu: int, shards: int, shard_id: int, batch: int, max_episodes: int | None):
    device = f"cuda:{gpu}"
    torch.cuda.set_device(gpu)
    common = load_common()
    cfg = load_tasks()[task]

    progress(f"[extract] {task} loading model on {device}")
    t0 = time.time()
    model, _ = load_vjepa2_vit_l(device)
    progress(f"[extract] {task} model loaded {time.time()-t0:.1f}s")

    eps_all = parquet_episode_ids(task)
    eps = [e for i, e in enumerate(eps_all) if i % shards == shard_id]
    if max_episodes is not None:
        eps = eps[:max_episodes]

    cached_A = set(list_cached_episodes(task, "A"))
    cached_B = set(list_cached_episodes(task, "B"))
    todo = [e for e in eps if (e not in cached_A) or (e not in cached_B)]

    progress(f"[extract] {task} shard {shard_id+1}/{shards} batch={batch}: total {len(eps)} todo {len(todo)}")

    n_win_total = 0
    t_start = time.time()
    for i, ep in enumerate(todo):
        d = extract_episode(model, task, ep, batch, device)
        if not d:
            progress(f"[extract] {task} ep {ep}: skipped (T<{WINDOW})")
            continue
        save_episode_features(task, "A", ep, d["feats_A"], d["t_last"])
        save_episode_features(task, "B", ep, d["feats_B"], d["t_last"])
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
    args = p.parse_args()
    run_task(
        args.task,
        gpu=args.gpu,
        shards=args.shards,
        shard_id=args.shard_id,
        batch=args.batch,
        max_episodes=args.max_episodes,
    )


if __name__ == "__main__":
    main()
