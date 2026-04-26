"""I/O helpers: configs, manifest, progress log, cache npz layout."""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROBE_ROOT = Path("/home/solee/physrepa_tasks/probe")
CONFIGS = PROBE_ROOT / "configs"
RESULTS = PROBE_ROOT / "results"
PROGRESS_PATH = RESULTS / "progress.md"
MANIFEST_PATH = PROBE_ROOT / "cache" / "manifest.json"


def load_common() -> dict[str, Any]:
    with open(CONFIGS / "_common.yaml") as f:
        return yaml.safe_load(f)


def load_tasks() -> dict[str, dict[str, Any]]:
    with open(CONFIGS / "tasks.yaml") as f:
        return yaml.safe_load(f)["tasks"]


def progress(line: str) -> None:
    """Append a timestamped line to results/progress.md."""
    stamp = _dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_PATH, "a") as f:
        f.write(f"\n[{stamp}] {line}\n")


def write_manifest(extra: dict[str, Any] | None = None) -> None:
    common = load_common()
    tasks = load_tasks()
    payload = {
        "spec": "/home/solee/physrepa_tasks/claude_code_task.md",
        "plan": "/root/.claude/plans/read-physrepa-tasks-claude-code-task-md-humming-planet.md",
        "dataset_root": common["dataset_root"],
        "vjepa2": common["vjepa2"],
        "window": common["window"],
        "variants_cached": common["variants"],
        "feature_dtype": common["feature_dtype"],
        "seed": common["seed"],
        "tasks": tasks,
        "weights_sha256": _sha256_path(common["vjepa2"]["weights_path"]),
    }
    if extra:
        payload.update(extra)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(payload, f, indent=2)


def _sha256_path(path: str, head_bytes: int = 1 << 20) -> str:
    """Hash the first MiB only — full hash on a 5 GB file is wasteful and we just
    need a stable identifier of the checkpoint variant."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(head_bytes))
    return f"head1MiB:{h.hexdigest()}"


def cache_paths(task: str, variant: str) -> Path:
    return PROBE_ROOT / "cache" / task / f"variant_{variant}"


def targets_path(task: str) -> Path:
    return PROBE_ROOT / "cache" / task / "targets.npz"


def save_targets(task: str, payload: dict[str, np.ndarray]) -> None:
    p = targets_path(task)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, **payload)


def load_targets(task: str) -> dict[str, np.ndarray]:
    p = targets_path(task)
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def save_episode_features(task: str, variant: str, episode_id: int, feats: np.ndarray, t_last: np.ndarray) -> None:
    """feats: [N_win, 24, D] float16; t_last: [N_win] int32."""
    p = cache_paths(task, variant) / f"episode_{episode_id:06d}.npz"
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        p,
        feats=feats.astype(np.float16, copy=False),
        episode_id=np.full((feats.shape[0],), episode_id, dtype=np.int32),
        t_last=t_last.astype(np.int32, copy=False),
    )


def load_episode_features(task: str, variant: str, episode_id: int) -> dict[str, np.ndarray]:
    p = cache_paths(task, variant) / f"episode_{episode_id:06d}.npz"
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def list_cached_episodes(task: str, variant: str) -> list[int]:
    p = cache_paths(task, variant)
    if not p.exists():
        return []
    return sorted(int(x.stem.split("_")[1]) for x in p.glob("episode_*.npz"))
