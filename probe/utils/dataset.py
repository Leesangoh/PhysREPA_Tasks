"""Dataset access: episode parquet iteration, MP4 frame decoding, window construction."""

from __future__ import annotations

import glob
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from .io import load_common, load_tasks


def parquet_paths(task: str) -> list[str]:
    common = load_common()
    return sorted(glob.glob(os.path.join(common["dataset_root"], task, "data", "chunk-*", "episode_*.parquet")))


def video_path(task: str, episode_id: int, camera: str = "image_0") -> str:
    """Find episode_<id>.mp4 across chunks (LeRobot-V2 layout)."""
    common = load_common()
    base = os.path.join(common["dataset_root"], task, "videos")
    pat = os.path.join(base, f"chunk-*/observation.images.{camera}/episode_{episode_id:06d}.mp4")
    hits = glob.glob(pat)
    if not hits:
        raise FileNotFoundError(pat)
    return hits[0]


def parquet_episode_ids(task: str) -> list[int]:
    return [int(Path(p).stem.split("_")[1]) for p in parquet_paths(task)]


def parquet_for_episode(task: str, episode_id: int) -> str:
    common = load_common()
    base = os.path.join(common["dataset_root"], task)
    pat = os.path.join(base, "data", "chunk-*", f"episode_{episode_id:06d}.parquet")
    hits = glob.glob(pat)
    if not hits:
        raise FileNotFoundError(pat)
    return hits[0]


def decode_mp4_rgb(path: str, expected_h: int = 384, expected_w: int = 384) -> np.ndarray:
    """Decode the entire MP4 to a [T, H, W, 3] uint8 array via ffmpeg subprocess."""
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True,
        check=True,
    )
    arr = np.frombuffer(proc.stdout, dtype=np.uint8).reshape(-1, expected_h, expected_w, 3)
    return arr


def windows_for_T(T: int, window_size: int = 16) -> np.ndarray:
    """Indices of last frame t_last for windows of size W with stride 1: t in {W-1 ... T-1}."""
    if T < window_size:
        return np.empty((0,), dtype=np.int64)
    return np.arange(window_size - 1, T, dtype=np.int64)


def task_dt(task: str) -> float:
    """Seconds per frame from per-task fps in tasks.yaml."""
    fps = load_tasks()[task]["fps"]
    return 1.0 / float(fps)


def task_object_keys(task: str) -> tuple[str, str, str | None] | None:
    """Return (pos_col, vel_col, acc_col_or_None) per task config; None if no object."""
    cfg = load_tasks()[task]
    if not cfg["has_object"]:
        return None
    pref = cfg["object_prefix"]  # "object", "handle", "peg", "nut"
    pos = f"physics_gt.{pref}_position"
    vel = f"physics_gt.{pref}_velocity"
    acc = f"physics_gt.{pref}_acceleration" if cfg["has_native_obj_acc"] else None
    return pos, vel, acc
