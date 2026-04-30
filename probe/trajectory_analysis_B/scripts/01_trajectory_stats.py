#!/usr/bin/env python3
"""Per-(task, layer, episode) trajectory geometry stats:
- path_length  : sum_t ||x_{t+1} - x_t||
- max_speed    : max_t ||x_{t+1} - x_t||
- mean_speed   : mean_t ||x_{t+1} - x_t||
- direct_dist  : ||x_{T-1} - x_0||
- tortuosity   : path_length / max(direct_dist, eps)
- curvature_mean: mean angular change between consecutive directions
- speed_profile: full speed sequence (saved per (task, layer) as percentile bands)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ta_utils.loader import ALL_TASKS, all_trajectories


OUT_STATS = ROOT / "results" / "stats"
OUT_STATS.mkdir(parents=True, exist_ok=True)


def trajectory_stats(X: np.ndarray) -> dict:
    """X: [T, D] one episode at one layer."""
    T = X.shape[0]
    if T < 2:
        return {"path_length": 0.0, "max_speed": 0.0, "mean_speed": 0.0,
                "direct_dist": 0.0, "tortuosity": 1.0, "curvature_mean": 0.0,
                "T": T}
    d = np.diff(X, axis=0)                     # [T-1, D]
    speeds = np.linalg.norm(d, axis=1)         # [T-1]
    path_length = float(speeds.sum())
    direct_dist = float(np.linalg.norm(X[-1] - X[0]))
    tortuosity = path_length / max(direct_dist, 1e-9)
    # Direction unit vectors and angular curvature
    units = d / np.clip(speeds[:, None], 1e-12, None)
    if units.shape[0] >= 2:
        cos_angle = np.clip((units[:-1] * units[1:]).sum(axis=1), -1.0, 1.0)
        angles = np.arccos(cos_angle)
        curvature_mean = float(angles.mean())
    else:
        curvature_mean = 0.0
    return {
        "path_length": path_length,
        "max_speed": float(speeds.max()),
        "mean_speed": float(speeds.mean()),
        "direct_dist": direct_dist,
        "tortuosity": tortuosity,
        "curvature_mean": curvature_mean,
        "T": T,
    }


def main():
    rows = []
    speed_profiles: dict[tuple[str, int], list[np.ndarray]] = {}

    for task in ALL_TASKS:
        print(f"[01_stats] {task}: loading trajectories...", flush=True)
        trajs = all_trajectories(task)
        print(f"[01_stats] {task}: {len(trajs)} episodes", flush=True)
        for traj in trajs:
            feats = traj["feats"]   # [T, 24, D]
            T = feats.shape[0]
            for L in range(24):
                X = feats[:, L, :]
                stats = trajectory_stats(X)
                stats["task"] = task
                stats["episode_id"] = int(traj["episode_id"])
                stats["layer"] = L
                rows.append(stats)
                # Speed profile: store sequence and aggregate later.
                if T >= 2:
                    d = np.diff(X, axis=0)
                    sp = np.linalg.norm(d, axis=1)
                    speed_profiles.setdefault((task, L), []).append(sp)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_STATS / "trajectory_stats.csv", index=False)
    print(f"[01_stats] wrote {OUT_STATS / 'trajectory_stats.csv'} ({len(df)} rows)", flush=True)

    # Aggregate by (task, layer)
    agg = (df.groupby(["task", "layer"])[["path_length", "max_speed", "mean_speed",
           "direct_dist", "tortuosity", "curvature_mean"]]
             .agg(["mean", "std"]))
    agg.columns = ["__".join(c) for c in agg.columns]
    agg = agg.reset_index()
    agg.to_csv(OUT_STATS / "trajectory_stats_summary.csv", index=False)
    print(f"[01_stats] wrote summary ({len(agg)} rows)", flush=True)

    # Speed profile percentiles per (task, layer)
    profile_rows = []
    for (task, L), seqs in speed_profiles.items():
        # Pad to common max length
        Tmax = max(s.size for s in seqs)
        padded = np.full((len(seqs), Tmax), np.nan)
        for i, s in enumerate(seqs):
            padded[i, : s.size] = s
        # Per-time-step percentiles across episodes
        for t in range(Tmax):
            vals = padded[:, t]
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            profile_rows.append({
                "task": task, "layer": L, "t": t,
                "p25": float(np.percentile(vals, 25)),
                "p50": float(np.percentile(vals, 50)),
                "p75": float(np.percentile(vals, 75)),
                "n_eps": int(vals.size),
            })
    pd.DataFrame(profile_rows).to_csv(OUT_STATS / "speed_profiles.csv", index=False)
    print(f"[01_stats] wrote speed profiles ({len(profile_rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
