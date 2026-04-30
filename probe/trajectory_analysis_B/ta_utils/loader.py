"""Sample episodes per task and load Variant B features (8192-d, 8 temporal slots × 1024).
Each trajectory is [N_win, 24, 8192] for one episode."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROBE_ROOT = Path("/home/solee/physrepa_tasks/probe")
sys.path.insert(0, str(PROBE_ROOT))

from utils.io import list_cached_episodes, load_episode_features, load_targets

ALL_TASKS = ["push", "strike", "reach", "drawer", "peg_insert", "nut_thread"]
SEED = 42
N_EPS_PER_TASK = 30
VARIANT = "B"


def sample_episodes(task: str, n: int = N_EPS_PER_TASK) -> list[int]:
    eps = list_cached_episodes(task, VARIANT)
    rng = np.random.default_rng(SEED)
    if len(eps) <= n:
        return eps
    pick = rng.choice(len(eps), size=n, replace=False)
    return [eps[i] for i in sorted(pick.tolist())]


def load_trajectory(task: str, episode_id: int) -> dict:
    """Returns {feats: [N_win, 24, 8192] fp32, t_last: [N_win], episode_id}."""
    d = load_episode_features(task, VARIANT, episode_id)
    feats = d["feats"].astype(np.float32, copy=False)
    return {
        "feats": feats,
        "t_last": d["t_last"],
        "episode_id": episode_id,
    }


def all_trajectories(task: str, n_eps: int = N_EPS_PER_TASK) -> list[dict]:
    return [load_trajectory(task, ep) for ep in sample_episodes(task, n_eps)]
