#!/usr/bin/env python3
"""Episode-level physics-parameter probing (mass, friction, restitution, damping).

Different methodology from window-level probing — physics params are CONSTANT
within an episode, so we use:
  - one (X, y) pair per episode: X = layer-mean of all-window features for that episode, y = scalar physics value
  - Ridge regression with leave-one-out or 5-fold episode CV
  - Per (task, layer, param) → R²

Per-task physics params (per CLAUDE.md):
  push:       mass, obj_friction, surface_friction
  strike:     mass, friction, surface_friction, restitution
  drawer:     drawer_joint_damping
  peg_insert: held_friction, fixed_friction, held_mass
  nut_thread: held_friction, fixed_friction, held_mass

Outputs:
- probe/results/<task>/physics_params/<param>.csv: layer × fold × R²
- probe/results/<task>/physics_params/_summary.csv: layer × param → R² mean/std
- probe/results/plots/physics_param_<task>.png: layer (x) × R² (y), one line per param
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
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

PROBE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROBE_ROOT))

from utils.io import list_cached_episodes, load_common, load_episode_features
from utils.dataset import parquet_for_episode


RESULTS = PROBE_ROOT / "results"
PLOTS = RESULTS / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

ALL_TASKS = ["push", "strike", "reach", "drawer", "peg_insert", "nut_thread"]


"""Physics parameters live in meta/episodes.jsonl (per-episode), NOT in parquet
per-frame data. Each task has different params:"""
PHYSICS_PARAMS = {
    "push":       ["object_0_mass", "object_0_static_friction", "object_0_dynamic_friction",
                   "surface_static_friction", "surface_dynamic_friction"],
    "strike":     ["object_0_mass", "object_0_static_friction", "object_0_dynamic_friction",
                   "surface_static_friction", "surface_dynamic_friction", "object_0_restitution"],
    "drawer":     ["drawer_joint_damping", "drawer_handle_mass",
                   "handle_static_friction", "handle_dynamic_friction"],
    "peg_insert": ["peg_static_friction", "peg_dynamic_friction", "peg_mass",
                   "hole_static_friction", "hole_dynamic_friction"],
    "nut_thread": ["nut_static_friction", "nut_dynamic_friction", "nut_mass",
                   "bolt_static_friction", "bolt_dynamic_friction"],
    "reach":      [],
}


def load_physics_param_per_episode(task: str, episode_ids: list[int]) -> tuple[np.ndarray, list[str]]:
    """Read physics params from meta/episodes.jsonl. Returns
    Y [n_eps, n_params_present], list_of_param_names_present."""
    import json
    expected = PHYSICS_PARAMS.get(task, [])
    if not expected:
        return np.zeros((len(episode_ids), 0)), []

    common = load_common()
    jsonl_path = Path(common["dataset_root"]) / task / "meta" / "episodes.jsonl"
    ep_to_phys: dict[int, dict[str, float]] = {}
    if jsonl_path.exists():
        with open(jsonl_path) as f:
            for line in f:
                d = json.loads(line)
                ep_to_phys[int(d["episode_index"])] = {
                    k: float(d[k]) for k in expected if k in d
                }
    rows = []
    for ep in episode_ids:
        rec = ep_to_phys.get(ep, {})
        rows.append({c: rec.get(c, np.nan) for c in expected})
    df_phys = pd.DataFrame(rows)
    keep_cols = [c for c in expected if df_phys[c].notna().any() and df_phys[c].std() > 1e-9]
    Y = df_phys[keep_cols].to_numpy(dtype=np.float32)
    return Y, keep_cols


def stack_episode_means(task: str, episode_ids: list[int], variant: str = "A") -> np.ndarray:
    """Returns X [n_eps, 24, D] — per-episode mean of window features per layer."""
    arrs = []
    for ep in episode_ids:
        d = load_episode_features(task, variant, ep)
        feats = d["feats"].astype(np.float32, copy=False)   # [N_win, 24, D]
        arrs.append(feats.mean(axis=0))                      # [24, D]
    return np.stack(arrs, axis=0)


def run_task(task: str, variant: str = "A", n_max_eps: int = 200, gpu: int = 0) -> None:
    expected = PHYSICS_PARAMS.get(task, [])
    if not expected:
        print(f"[07_physics] {task}: no physics params declared, skipping")
        return

    eps = list_cached_episodes(task, variant)
    if not eps:
        print(f"[07_physics] {task}: no cached features, skipping")
        return
    rng = np.random.default_rng(42)
    if len(eps) > n_max_eps:
        eps = sorted(rng.choice(eps, size=n_max_eps, replace=False).tolist())
    print(f"[07_physics] {task}: using {len(eps)} episodes", flush=True)

    Y, params = load_physics_param_per_episode(task, eps)
    if Y.shape[1] == 0:
        print(f"[07_physics] {task}: no varying physics params present, skipping")
        return
    print(f"[07_physics] {task}: params with variation = {params}", flush=True)

    # Drop episodes with any NaN physics
    ok = np.isfinite(Y).all(axis=1)
    if ok.sum() < len(eps):
        print(f"[07_physics] {task}: {(~ok).sum()} eps with NaN physics — dropped")
    Y = Y[ok]; eps_use = [eps[i] for i in range(len(eps)) if ok[i]]

    print(f"[07_physics] {task}: loading episode-mean features...", flush=True)
    X = stack_episode_means(task, eps_use, variant=variant)        # [n_eps, 24, D]
    print(f"[07_physics] {task}: X {X.shape}, Y {Y.shape}", flush=True)

    out_dir = RESULTS / task / "physics_params"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    summary_rows = []
    n_splits = min(5, X.shape[0])
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for L in range(X.shape[1]):
        XL = X[:, L, :]
        for j, param in enumerate(params):
            y = Y[:, j]
            r2_folds = []
            for fold, (tr, te) in enumerate(kf.split(XL)):
                X_tr, X_te = XL[tr], XL[te]
                y_tr, y_te = y[tr], y[te]
                # Standardize within fold
                mu_x = X_tr.mean(0); sd_x = X_tr.std(0) + 1e-9
                X_tr_n = (X_tr - mu_x) / sd_x
                X_te_n = (X_te - mu_x) / sd_x
                mu_y = y_tr.mean(); sd_y = y_tr.std() + 1e-9
                y_tr_n = (y_tr - mu_y) / sd_y
                # Ridge
                model = Ridge(alpha=1.0)
                model.fit(X_tr_n, y_tr_n)
                pred = model.predict(X_te_n) * sd_y + mu_y
                r2 = float(r2_score(y_te, pred))
                rows.append({"task": task, "layer": L, "param": param, "fold": fold, "r2": r2,
                             "n_train_eps": len(tr), "n_test_eps": len(te)})
                r2_folds.append(r2)
            summary_rows.append({"task": task, "layer": L, "param": param,
                                 "r2_mean": float(np.mean(r2_folds)),
                                 "r2_std": float(np.std(r2_folds))})

    pd.DataFrame(rows).to_csv(out_dir / "_per_fold.csv", index=False)
    sumdf = pd.DataFrame(summary_rows)
    sumdf.to_csv(out_dir / "_summary.csv", index=False)
    print(f"[07_physics] {task}: wrote summary ({len(sumdf)} rows)", flush=True)

    # Plot: layer (x) vs R² (y), one line per param
    fig, ax = plt.subplots(figsize=(9, 5))
    for param in sumdf.param.unique():
        sub = sumdf[sumdf.param == param].sort_values("layer")
        ax.plot(sub.layer, sub.r2_mean, "-o", markersize=4, label=param.replace("physics_gt.", ""))
        ax.fill_between(sub.layer, sub.r2_mean - sub.r2_std, sub.r2_mean + sub.r2_std, alpha=0.15)
    ax.set_xlabel("Layer")
    ax.set_ylabel("R² (Ridge, episode-level 5-fold CV)")
    ax.set_title(f"Physics-parameter probing per layer — {task} (Variant {variant})")
    ax.set_xlim(-0.5, 23.5)
    ax.axhline(0, ls=":", color="gray")
    ax.axvspan(6, 18, alpha=0.05, color="green")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS / f"physics_param_{task}_{variant}.png", dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="all", help="all or comma-separated")
    p.add_argument("--variant", default="A")
    p.add_argument("--n-max-eps", type=int, default=200)
    args = p.parse_args()
    tasks = ALL_TASKS if args.task == "all" else args.task.split(",")
    for task in tasks:
        run_task(task, variant=args.variant, n_max_eps=args.n_max_eps)


if __name__ == "__main__":
    main()
