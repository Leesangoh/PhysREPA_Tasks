#!/usr/bin/env python3
"""Phase-space latent geometry on Variant A trajectories.

Analyze per-layer geometry in the latent phase-space formed by:
  - z_t   : latent state at window t
  - dz_t  : z_{t+1} - z_t

For each (task, layer), and for contact-varying tasks additionally each split
in {pre, during, post}, compute:
  - mean/std of ||dz||
  - curvature kappa in the 2D phase-space (PC1(z), PC1(dz))
  - sweep volume as the 95th-percentile bounding ellipse area
  - local Lyapunov-style estimate on dz differences

Outputs:
  - trajectory_analysis_B/results/stats/phase_space.csv
  - trajectory_analysis_B/results/plots/phase_space_<task>.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROBE_ROOT = Path("/home/solee/physrepa_tasks/probe")
sys.path.insert(0, str(PROBE_ROOT))

from utils.dataset import parquet_for_episode
from utils.io import list_cached_episodes, load_episode_features


STATS_DIR = ROOT / "results" / "stats"
PLOTS_DIR = ROOT / "results" / "plots"
LOGS_DIR = ROOT / "results" / "logs"
STATS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

ALL_TASKS = ["push", "strike", "reach", "drawer", "peg_insert", "nut_thread"]
TASKS_CONTACT_VARYING = ["push", "strike", "drawer"]
SPLITS_ALL = ["all"]
SPLITS_PHASE3 = ["all", "pre", "during", "post"]
SEED = 42
DEFAULT_N_EPS = 30
EPS = 1e-9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Variant A phase-space geometry.")
    parser.add_argument("--task", default="all", help="all or comma-separated task list")
    parser.add_argument("--n-episodes", type=int, default=DEFAULT_N_EPS)
    parser.add_argument("--dry-run", action="store_true", help="push only, layers 0/11/23, 6 episodes")
    return parser.parse_args()


def sample_episodes(task: str, n: int) -> list[int]:
    eps = list_cached_episodes(task, "A")
    rng = np.random.default_rng(SEED)
    if len(eps) <= n:
        return eps
    idx = rng.choice(len(eps), size=n, replace=False)
    return [eps[i] for i in sorted(idx.tolist())]


def contact_phase3_labels(task: str, episode_id: int, t_last: np.ndarray) -> np.ndarray:
    """Return per-window labels in {'pre','during','post'}.

    Episodes with no contact are labeled entirely 'pre'.
    Within [first_contact, last_contact], transient zeros are conservatively
    folded into 'during' to avoid sparse interior gaps.
    """
    labels = np.full(t_last.size, "pre", dtype=object)
    if task not in TASKS_CONTACT_VARYING:
        return labels
    try:
        df = pq.read_table(parquet_for_episode(task, episode_id), columns=["physics_gt.contact_flag"]).to_pandas()
        cf = np.array(
            [
                float(v) if not isinstance(v, np.ndarray) else float(np.asarray(v).reshape(-1)[0])
                for v in df["physics_gt.contact_flag"].tolist()
            ],
            dtype=np.float32,
        )
    except Exception:
        return labels

    cf_idx = np.where(cf == 1.0)[0]
    if cf_idx.size == 0:
        return labels

    first = int(cf_idx[0])
    last = int(cf_idx[-1])
    for i, t in enumerate(t_last):
        t_int = int(t)
        if t_int < first:
            labels[i] = "pre"
        elif t_int > last:
            labels[i] = "post"
        else:
            labels[i] = "during"
    return labels


def project_pc1(arr: np.ndarray) -> np.ndarray:
    """Project [N, D] onto the first principal component."""
    if arr.shape[0] < 2:
        return np.full((arr.shape[0],), np.nan, dtype=np.float32)
    pca = PCA(n_components=1, random_state=SEED)
    return pca.fit_transform(arr.astype(np.float32, copy=False)).reshape(-1).astype(np.float32)


def curvature_kappa(points_2d: np.ndarray) -> float:
    """Angle-change sum / arc length over an ordered 2D trajectory."""
    if points_2d.shape[0] < 3:
        return float("nan")
    steps = np.diff(points_2d, axis=0)
    norms = np.linalg.norm(steps, axis=1)
    if steps.shape[0] < 2 or float(norms.sum()) <= EPS:
        return float("nan")
    total_turn = 0.0
    for i in range(steps.shape[0] - 1):
        den = float(norms[i] * norms[i + 1] + EPS)
        if den <= EPS:
            continue
        cos = float(np.clip(np.dot(steps[i], steps[i + 1]) / den, -1.0, 1.0))
        total_turn += float(np.arccos(cos))
    return float(total_turn / max(float(norms.sum()), EPS))


def sweep_volume(points_2d: np.ndarray) -> float:
    """95th-percentile axis-aligned ellipse area in 2D."""
    if points_2d.shape[0] < 2:
        return float("nan")
    center = np.nanmedian(points_2d, axis=0)
    dev = np.abs(points_2d - center[None, :])
    rx = float(np.nanquantile(dev[:, 0], 0.95))
    ry = float(np.nanquantile(dev[:, 1], 0.95))
    return float(np.pi * rx * ry)


def lyap_local(dz_seq: np.ndarray) -> float:
    """Local Lyapunov-like statistic on successive dz changes."""
    if dz_seq.shape[0] < 3:
        return float("nan")
    diffs = np.linalg.norm(np.diff(dz_seq, axis=0), axis=1)
    diffs = np.clip(diffs.astype(np.float64, copy=False), EPS, None)
    if diffs.size < 2:
        return float("nan")
    return float(np.mean(np.log(diffs[1:]) - np.log(diffs[:-1])))


def collect_episode_payload(task: str, episode_id: int) -> dict | None:
    try:
        d = load_episode_features(task, "A", episode_id)
        feats = d["feats"].astype(np.float32, copy=False)  # [T, 24, 1024]
        t_last = d["t_last"].astype(np.int64, copy=False)
        if feats.shape[0] < 2:
            return None
        labels = contact_phase3_labels(task, episode_id, t_last)
        return {
            "episode_id": episode_id,
            "feats": feats,
            "t_last": t_last,
            "labels": labels,
        }
    except Exception as exc:
        with open(LOGS_DIR / "18_phase_space_geometry.log", "a") as f:
            f.write(f"[{task} ep={episode_id}] load failed: {exc}\n")
        return None


def summarize_split(layer_payloads: list[dict], split_name: str) -> dict[str, float]:
    """Aggregate per-episode stats for one layer/split."""
    dz_norm_all: list[np.ndarray] = []
    episode_curv: list[float] = []
    episode_lyap: list[float] = []
    all_proj_points: list[np.ndarray] = []
    n_pairs_total = 0
    n_eps_used = 0

    # First pass: collect all z and dz points used by this split for PCA bases.
    z_pool = []
    dz_pool = []
    per_episode = []
    for payload in layer_payloads:
        Z = payload["Z"]
        labels = payload["labels"]
        pair_labels = labels[:-1]
        mask = np.ones(pair_labels.shape[0], dtype=bool) if split_name == "all" else (pair_labels == split_name)
        if mask.sum() < 2:
            continue
        z_sel = Z[:-1][mask]
        dz_sel = np.diff(Z, axis=0)[mask]
        z_pool.append(z_sel)
        dz_pool.append(dz_sel)
        per_episode.append((z_sel, dz_sel))

    if not per_episode:
        return {
            "n_episodes": 0,
            "n_pairs": 0,
            "mean_dz_norm": float("nan"),
            "std_dz_norm": float("nan"),
            "curvature_kappa": float("nan"),
            "sweep_volume": float("nan"),
            "lyap_local": float("nan"),
        }

    z_basis = project_pc1(np.concatenate(z_pool, axis=0))
    dz_basis = project_pc1(np.concatenate(dz_pool, axis=0))

    # Refit PCA objects to project episode-level points consistently.
    z_pca = PCA(n_components=1, random_state=SEED)
    dz_pca = PCA(n_components=1, random_state=SEED)
    z_pca.fit(np.concatenate(z_pool, axis=0))
    dz_pca.fit(np.concatenate(dz_pool, axis=0))

    for z_sel, dz_sel in per_episode:
        dz_norm = np.linalg.norm(dz_sel, axis=1)
        dz_norm_all.append(dz_norm)
        n_pairs_total += int(dz_sel.shape[0])
        n_eps_used += 1

        z1 = z_pca.transform(z_sel).reshape(-1)
        dz1 = dz_pca.transform(dz_sel).reshape(-1)
        points = np.stack([z1, dz1], axis=1)
        all_proj_points.append(points)
        episode_curv.append(curvature_kappa(points))
        episode_lyap.append(lyap_local(dz_sel))

    dz_concat = np.concatenate(dz_norm_all, axis=0) if dz_norm_all else np.empty((0,), dtype=np.float32)
    pts_concat = np.concatenate(all_proj_points, axis=0) if all_proj_points else np.empty((0, 2), dtype=np.float32)

    return {
        "n_episodes": int(n_eps_used),
        "n_pairs": int(n_pairs_total),
        "mean_dz_norm": float(np.nanmean(dz_concat)) if dz_concat.size else float("nan"),
        "std_dz_norm": float(np.nanstd(dz_concat)) if dz_concat.size else float("nan"),
        "curvature_kappa": float(np.nanmean(np.asarray(episode_curv, dtype=np.float64))) if episode_curv else float("nan"),
        "sweep_volume": sweep_volume(pts_concat),
        "lyap_local": float(np.nanmean(np.asarray(episode_lyap, dtype=np.float64))) if episode_lyap else float("nan"),
    }


def main() -> None:
    args = parse_args()
    tasks = ["push"] if args.dry_run else (ALL_TASKS if args.task == "all" else [x.strip() for x in args.task.split(",") if x.strip()])
    n_episodes = 6 if args.dry_run else args.n_episodes
    layers = [0, 11, 23] if args.dry_run else list(range(24))

    rows: list[dict] = []
    for task in tasks:
        print(f"[18_phase_space] {task}: sampling episodes", flush=True)
        eps = sample_episodes(task, n_episodes)
        episode_payloads = []
        for ep in eps:
            payload = collect_episode_payload(task, ep)
            if payload is not None:
                episode_payloads.append(payload)
        print(f"[18_phase_space] {task}: kept {len(episode_payloads)}/{len(eps)} episodes", flush=True)

        for layer in layers:
            try:
                layer_payloads = [
                    {
                        "episode_id": p["episode_id"],
                        "Z": p["feats"][:, layer, :],
                        "labels": p["labels"],
                    }
                    for p in episode_payloads
                ]
                split_names = SPLITS_PHASE3 if task in TASKS_CONTACT_VARYING else SPLITS_ALL
                for split_name in split_names:
                    stats = summarize_split(layer_payloads, split_name)
                    rows.append(
                        {
                            "task": task,
                            "layer": int(layer),
                            "target_split": split_name,
                            **stats,
                        }
                    )
            except Exception as exc:
                with open(LOGS_DIR / "18_phase_space_geometry.log", "a") as f:
                    f.write(f"[{task} layer={layer}] summarize failed: {exc}\n")
                continue

    df = pd.DataFrame(rows)
    out_csv = STATS_DIR / "phase_space.csv"
    if not df.empty:
        df = df[
            [
                "task",
                "layer",
                "target_split",
                "n_episodes",
                "n_pairs",
                "mean_dz_norm",
                "std_dz_norm",
                "curvature_kappa",
                "sweep_volume",
                "lyap_local",
            ]
        ]
    df.to_csv(out_csv, index=False)
    print(f"[18_phase_space] wrote {len(df)} rows -> {out_csv}", flush=True)

    if args.dry_run:
        if not df.empty:
            print(df.to_string(index=False), flush=True)
        return

    for task in tasks:
        sub = df[df["task"] == task].copy()
        if sub.empty:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), squeeze=False)
        metrics = ["mean_dz_norm", "curvature_kappa", "sweep_volume"]
        split_order = SPLITS_PHASE3 if task in TASKS_CONTACT_VARYING else SPLITS_ALL
        color_map = {"all": "black", "pre": "#1f77b4", "during": "#d62728", "post": "#2ca02c"}
        for ax, metric in zip(axes.flatten(), metrics):
            for split_name in split_order:
                sdf = sub[sub["target_split"] == split_name].sort_values("layer")
                if sdf.empty:
                    continue
                style = "--" if split_name == "all" else "-"
                ax.plot(
                    sdf["layer"].to_numpy(),
                    sdf[metric].to_numpy(),
                    linestyle=style,
                    color=color_map.get(split_name, None),
                    linewidth=1.5,
                    label=split_name,
                )
            ax.set_xlabel("Layer")
            ax.set_ylabel(metric)
            ax.set_title(f"{metric} — {task}")
            ax.set_xlim(-0.5, 23.5)
            ax.grid(alpha=0.3)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4))
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(PLOTS_DIR / f"phase_space_{task}.png", dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    main()
