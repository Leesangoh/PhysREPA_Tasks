#!/usr/bin/env python3
"""Phase 1c — Time-only baseline (Ridge probe with feature = normalized window time).

For "leakage-prone" targets that may be predictable from clip time alone (e.g. a
script-driven \texttt{phase} integer, monotonic progress scalars, drawer joint
position climbing through the episode), we run a 1-D Ridge probe with the
single feature

    x_i = t_last_i / max_t_last_for_episode_i

per window. This isolates how much of the apparent decodability comes from
"model knows physics" vs "model just knows clip time". A high R^2 here is
a *warning*: any V-JEPA layer that beats this number by less than a small
margin is likely riding on time leakage rather than physics representation.

Targets (limited to those that exist for the task):
    phase                          (all six tasks; constant for several)
    axial_progress                 (nut_thread)
    insertion_depth                (peg_insert)
    peg_hole_lateral_error         (peg_insert)
    drawer_joint_pos               (drawer)
    drawer_joint_vel               (drawer)
    drawer_opening_extent          (drawer)

Output:
    probe/results/time_only_baseline.csv
    cols: task, target, fold, n_train_windows, n_test_windows, r2, mse

Within-fold standardization on inner-train stats only. 5-fold GroupKFold by
\texttt{episode_id}; this matches the main-probe protocol so the numbers are
directly comparable to the V-JEPA Variant A R^2 published in
results/<task>/variant_A/<target>.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

PROBE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROBE_ROOT))

from utils.io import load_targets, progress  # noqa: E402

RESULTS = PROBE_ROOT / "results"
OUT_CSV = RESULTS / "time_only_baseline.csv"

TASK_TARGETS: dict[str, list[str]] = {
    "push":       ["phase"],
    "strike":     ["phase"],
    "reach":      ["phase"],
    "drawer":     ["phase", "drawer_joint_pos", "drawer_joint_vel", "drawer_opening_extent"],
    "peg_insert": ["phase", "insertion_depth", "peg_hole_lateral_error"],
    "nut_thread": ["phase", "axial_progress"],
}

N_SPLITS = 5
SEED = 42


def normalized_time_feature(episode_id: np.ndarray, t_last: np.ndarray) -> np.ndarray:
    """Compute t / T_max per episode. Returns shape (N, 1) float32.

    For each episode, T_max is that episode's largest t_last across windows,
    so the feature falls in [(t_min/T_max) ... 1.0].
    """
    out = np.zeros(t_last.size, dtype=np.float32)
    uniq = np.unique(episode_id)
    for ep in uniq:
        m = episode_id == ep
        tmax = float(t_last[m].max())
        if tmax <= 0:
            tmax = 1.0
        out[m] = t_last[m].astype(np.float32) / tmax
    return out.reshape(-1, 1)


def fit_eval_fold(X_tr, y_tr, X_te, y_te) -> tuple[float, float]:
    """Within-fold z-score using inner-train stats; Ridge alpha=1.

    Returns (r2, mse).
    """
    mu_x = X_tr.mean(0); sd_x = X_tr.std(0) + 1e-9
    mu_y = y_tr.mean(); sd_y = y_tr.std() + 1e-9
    X_tr_n = (X_tr - mu_x) / sd_x
    X_te_n = (X_te - mu_x) / sd_x
    y_tr_n = (y_tr - mu_y) / sd_y

    m = Ridge(alpha=1.0)
    m.fit(X_tr_n, y_tr_n)
    pred = m.predict(X_te_n) * sd_y + mu_y
    r2 = float(r2_score(y_te, pred))
    mse = float(np.mean((pred - y_te) ** 2))
    return r2, mse


def run_target(task: str, target: str, tgt: dict[str, np.ndarray]) -> list[dict]:
    if target not in tgt:
        return []
    y = tgt[target]
    # Targets cache stores scalar physics fields as (N, 1) — squeeze trailing axes
    # of size 1; reject anything with multi-D output (vector / quaternion).
    y = np.squeeze(y)
    if y.ndim != 1:
        return []
    eps = tgt["episode_id"]
    t_last = tgt["t_last"]
    X = normalized_time_feature(eps, t_last)

    finite = np.isfinite(y)
    if finite.sum() < 100:
        return []
    X = X[finite]; y = y[finite]; eps = eps[finite]
    if np.unique(eps).size < N_SPLITS:
        return []

    if y.std() < 1e-9:
        # Degenerate constant target — explicitly tagged in the CSV so the
        # downstream paper notes excludes from cross-task aggregates.
        return [{
            "task": task, "target": target, "fold": -1,
            "n_train_windows": int(X.shape[0]), "n_test_windows": 0,
            "r2": np.nan, "mse": 0.0, "note": "constant_target",
        }]

    gkf = GroupKFold(n_splits=N_SPLITS)
    rows = []
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups=eps)):
        if tr.size < 20 or te.size < 5:
            continue
        r2, mse = fit_eval_fold(X[tr], y[tr], X[te], y[te])
        rows.append({
            "task": task, "target": target, "fold": fold,
            "n_train_windows": int(tr.size), "n_test_windows": int(te.size),
            "r2": r2, "mse": mse, "note": "ok",
        })
    return rows


def main():
    rows: list[dict] = []
    for task, targets in TASK_TARGETS.items():
        try:
            tgt = load_targets(task)
        except FileNotFoundError:
            print(f"[time_only] {task}: targets cache missing — skip", flush=True)
            continue
        present = [t for t in targets if t in tgt]
        print(f"[time_only] {task}: targets in cache = {present}", flush=True)
        for target in targets:
            rs = run_target(task, target, tgt)
            rows.extend(rs)

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    progress(f"[time_only] wrote {len(df)} rows to {OUT_CSV}")
    print(f"[time_only] wrote {len(df)} rows -> {OUT_CSV}", flush=True)
    if not df.empty:
        agg = df[df.note == "ok"].groupby(["task", "target"]).r2.agg(["mean", "std", "count"]).round(3)
        print(agg.to_string(), flush=True)


if __name__ == "__main__":
    main()
