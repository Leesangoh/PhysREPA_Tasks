#!/usr/bin/env python3
"""Phase 4 — R3M (ResNet50) per-window feature extraction.

For each (task, episode):
  - decode MP4 → 384x384 RGB frames
  - resize to 224x224, ImageNet normalize
  - per-frame ResNet50 forward
  - hooks capture 5 intermediate stages: conv1+maxpool, layer1, layer2, layer3, layer4
  - per-stage spatial GAP → 16-frame window mean-pool
  - save npz keyed by stage_0..stage_4

Schema (per Codex review): one .npz per episode at
    /mnt/md1/solee/physprobe_features/<task>/r3m/episode_<id>.npz
with keys
    stage_0: float16 [N_win, 64]
    stage_1: float16 [N_win, 256]
    stage_2: float16 [N_win, 512]
    stage_3: float16 [N_win, 1024]
    stage_4: float16 [N_win, 2048]
    t_last:  int32  [N_win]
    episode_id: int32 [N_win]   (broadcast for join with targets)

This matches V-JEPA Variant A semantics (one feature per window) but with
5 layer-equivalent depth slots instead of 24. R3M is positioned as a static
image-encoder baseline per Codex Decision B (paper-claim ordering).

Usage:
    /isaac-sim/python.sh probe/scripts/11_extract_r3m_features.py \
        --task push --gpu 0 --batch 32 [--max-episodes N] [--dry-run]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROBE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROBE_ROOT))

from utils.dataset import decode_mp4_rgb, parquet_episode_ids, video_path  # noqa: E402
from utils.io import progress  # noqa: E402

WINDOW = 16
CACHE_ROOT = Path("/mnt/md1/solee/physprobe_features")
STAGE_DIMS = {"stage_0": 64, "stage_1": 256, "stage_2": 512, "stage_3": 1024, "stage_4": 2048}

# ImageNet stats (R3M's preprocessing)
IM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IM_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_r3m_resnet50(device: str) -> torch.nn.Module:
    """Load R3M's ResNet50 wrapper. Returns the inner ResNet50 for feature
    extraction (R3M itself wraps it with extra heads we don't need)."""
    from r3m import load_r3m

    r3m_module = load_r3m("resnet50")        # downloads on first use
    r3m_module.to(device)
    r3m_module.eval()
    # R3M.module is a torchvision ResNet50 with a 2048-d output.
    inner = r3m_module.module.convnet         # The bare ResNet50.
    inner.eval()
    return inner.to(device)


class StageHooks:
    """Forward hooks on a torchvision ResNet50 for 5-stage feature extraction.

    Uses post-pooling spatial outputs:
        stage_0: after maxpool        (B, 64, 56, 56) → GAP → (B, 64)
        stage_1: after layer1         (B, 256, 56, 56) → (B, 256)
        stage_2: after layer2         (B, 512, 28, 28) → (B, 512)
        stage_3: after layer3         (B, 1024, 14, 14) → (B, 1024)
        stage_4: after layer4         (B, 2048, 7, 7)   → (B, 2048)
    """

    def __init__(self, model: torch.nn.Module):
        self.outs: dict[str, torch.Tensor] = {}
        self.handles = []
        # ResNet50 attributes: conv1, bn1, relu, maxpool, layer1..layer4, avgpool, fc.
        self.handles.append(model.maxpool.register_forward_hook(self._make("stage_0")))
        self.handles.append(model.layer1.register_forward_hook(self._make("stage_1")))
        self.handles.append(model.layer2.register_forward_hook(self._make("stage_2")))
        self.handles.append(model.layer3.register_forward_hook(self._make("stage_3")))
        self.handles.append(model.layer4.register_forward_hook(self._make("stage_4")))

    def _make(self, name: str):
        def hook(_mod, _inp, out):
            # out: (B, C, H, W) — spatial GAP to (B, C)
            gap = out.float().mean(dim=(2, 3))
            self.outs[name] = gap
        return hook

    def collect(self) -> dict[str, torch.Tensor]:
        d = {k: v.clone() for k, v in self.outs.items()}
        self.outs.clear()
        return d

    def close(self):
        for h in self.handles:
            h.remove()


def windows_for_clip(x_full: torch.Tensor, batch_frames: int):
    """Generator over windows; yields (window_start, frames_block).

    x_full: (T, 3, 224, 224)
    Each window is the slice [t:t+16]. We process all 16 frames as one mini-batch
    of size 16 through ResNet50, then pool the 5 stages over time.
    """
    T = x_full.shape[0]
    if T < WINDOW:
        return
    n_win = T - WINDOW + 1
    # We forward 16 frames per window in a single batch. To amortize, we batch
    # `batch_frames` frames at a time in the underlying ResNet50 forward.
    for s in range(n_win):
        block = x_full[s:s + WINDOW]   # (16, 3, 224, 224)
        yield s, block


def preprocess_frames_r3m(frames_uint8: torch.Tensor, device: str) -> torch.Tensor:
    """Convert (T, H, W, 3) uint8 → (T, 3, 224, 224) float32, ImageNet normalized."""
    # to (T, 3, H, W) float [0,1]
    x = frames_uint8.permute(0, 3, 1, 2).float() / 255.0
    # resize to 224 with bilinear
    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    x = x.to(device)
    x = (x - IM_MEAN.to(device)) / IM_STD.to(device)
    return x


def extract_episode(
    model: torch.nn.Module,
    hooks: StageHooks,
    task: str,
    episode_id: int,
    device: str,
    *,
    fwd_batch: int = 64,
) -> dict[str, np.ndarray] | None:
    arr = decode_mp4_rgb(video_path(task, episode_id))   # (T, H, W, 3) uint8
    T = arr.shape[0]
    n_win = T - WINDOW + 1
    if n_win <= 0:
        return None
    frames = torch.from_numpy(arr.copy())           # (T, H, W, 3)
    x_full = preprocess_frames_r3m(frames, device)  # (T, 3, 224, 224) fp32 on GPU

    # Forward all T frames in batches of fwd_batch through ResNet50; collect 5
    # stages. Per-frame stage features are then mean-pooled over each window.
    stage_per_frame: dict[str, torch.Tensor] = {k: torch.zeros(T, d, device=device, dtype=torch.float32)
                                                for k, d in STAGE_DIMS.items()}
    with torch.inference_mode():
        for s in range(0, T, fwd_batch):
            e = min(s + fwd_batch, T)
            _ = model(x_full[s:e])           # triggers hooks
            d = hooks.collect()              # {stage_k: (B_b, dim_k)}
            for k, v in d.items():
                stage_per_frame[k][s:e] = v

    # Window mean-pool: for each window starting at t in [0, n_win), pool 16
    # consecutive per-frame features. Use cumulative trick for speed.
    out: dict[str, np.ndarray] = {}
    for k, full in stage_per_frame.items():
        # Cumulative sum along time, shape (T+1, dim)
        c = torch.cat([torch.zeros(1, full.shape[1], device=device, dtype=full.dtype), full.cumsum(dim=0)], dim=0)
        # window means: (c[t+16] - c[t]) / 16 for t in 0..n_win-1
        win_sum = c[WINDOW:WINDOW + n_win] - c[:n_win]
        win_mean = win_sum / WINDOW
        out[k] = win_mean.to(torch.float16).cpu().numpy()

    out["t_last"] = np.arange(WINDOW - 1, T, dtype=np.int32)
    out["episode_id"] = np.full((n_win,), int(episode_id), dtype=np.int32)
    return out


def save_episode(task: str, episode_id: int, payload: dict[str, np.ndarray]) -> None:
    out_dir = CACHE_ROOT / task / "r3m"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / f"episode_{episode_id:06d}.npz", **payload)


def list_cached_r3m(task: str) -> list[int]:
    d = CACHE_ROOT / task / "r3m"
    if not d.exists():
        return []
    out = []
    for p in d.glob("episode_*.npz"):
        try:
            out.append(int(p.stem.split("_", 1)[1]))
        except Exception:
            continue
    return sorted(out)


def run_task(
    task: str,
    *,
    gpu: int,
    batch: int,
    max_episodes: int | None,
    dry_run: bool,
) -> None:
    device = f"cuda:{gpu}"
    torch.cuda.set_device(gpu)

    progress(f"[r3m_extract] {task} loading R3M ResNet50 on {device}")
    t0 = time.time()
    model = load_r3m_resnet50(device)
    hooks = StageHooks(model)
    progress(f"[r3m_extract] {task} model loaded {time.time()-t0:.1f}s")

    eps = parquet_episode_ids(task)
    cached = set(list_cached_r3m(task))
    todo = [e for e in eps if e not in cached]
    if max_episodes is not None:
        todo = todo[:max_episodes]
    if dry_run:
        todo = todo[:3]
    progress(f"[r3m_extract] {task}: total {len(eps)} cached {len(cached)} todo {len(todo)}")

    n_win_total = 0
    t_start = time.time()
    for i, ep in enumerate(todo):
        d = extract_episode(model, hooks, task, ep, device, fwd_batch=batch)
        if d is None:
            progress(f"[r3m_extract] {task} ep {ep}: skipped (T<{WINDOW})")
            continue
        if dry_run:
            print(f"[r3m_extract:dry] ep={ep} n_win={d['t_last'].size} stage shapes:",
                  {k: tuple(v.shape) for k, v in d.items() if k.startswith('stage_')}, flush=True)
        else:
            save_episode(task, ep, d)
        n_win_total += d["t_last"].size

        if (i + 1) % 25 == 0 or (i + 1) == len(todo):
            dt = time.time() - t_start
            wps = n_win_total / max(dt, 1e-6)
            eta = (len(todo) - i - 1) * dt / max(i + 1, 1)
            progress(f"[r3m_extract] {task} {i+1}/{len(todo)} eps {n_win_total} win "
                     f"{wps:.1f} win/s ETA {eta/60:.1f}min")

    hooks.close()
    progress(f"[r3m_extract] {task} DONE: {len(todo)} eps {n_win_total} win in {(time.time()-t_start)/60:.1f}min")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--batch", type=int, default=64, help="frames per forward batch")
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument("--dry-run", action="store_true",
                   help="3-episode sanity print, no cache write")
    args = p.parse_args()
    run_task(
        args.task,
        gpu=args.gpu,
        batch=args.batch,
        max_episodes=args.max_episodes,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
