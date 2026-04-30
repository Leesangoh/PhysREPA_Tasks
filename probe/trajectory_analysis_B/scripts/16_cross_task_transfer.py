#!/usr/bin/env python3
"""Cross-task transfer probing (Codex 2nd recommendation).

Train probe on task A, test on task B. Compare to within-task baseline.
"Transfer gap" = within-task R² − transfer R². Small gap = task-general physics.

Targets: matched-units kinematics + contact:
  - ee_velocity (3D, m/s)
  - ee_acceleration (3D, m/s²)
  - contact_flag (binary {0,1})
  - contact_force_log1p_mag (1D, log1p N)

Pairs: bidirectional (push↔strike), plus drawer→{push,strike} and back.
Optional: leave-one-task-out using {push, strike, drawer} pool for contact subset.

Variant: A (1024-d). Cheaper and matches earlier kinematic results to compare against.

Outputs:
- results/stats/cross_task_transfer.csv:
    src_task, tgt_task, layer, target, transfer_r2, within_tgt_r2, gap
- results/plots/cross_task_transfer_<target>.png: layer × scheme heatmap
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

PROBE_ROOT = Path("/home/solee/physrepa_tasks/probe")
sys.path.insert(0, str(PROBE_ROOT))
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Use Variant A loader (override the B default)
from utils.io import list_cached_episodes, load_episode_features, load_targets


PLOTS = ROOT / "results" / "plots"
STATS = ROOT / "results" / "stats"
PLOTS.mkdir(parents=True, exist_ok=True)
STATS.mkdir(parents=True, exist_ok=True)


N_EPS = 50
TASKS = ["push", "strike", "drawer"]
PAIRS = [("push", "strike"), ("strike", "push"),
         ("push", "drawer"), ("drawer", "push"),
         ("strike", "drawer"), ("drawer", "strike")]
TARGETS = ["ee_velocity", "ee_acceleration", "contact_flag", "contact_force_log1p_mag"]


def load_task_arrays(task: str, n_eps: int = N_EPS, variant: str = "A"):
    eps = list_cached_episodes(task, variant)
    rng = np.random.default_rng(42)
    if len(eps) > n_eps:
        eps = sorted(rng.choice(eps, size=n_eps, replace=False).tolist())
    feats_list, eps_arr_list, t_last_list = [], [], []
    for ep in eps:
        d = load_episode_features(task, variant, ep)
        feats_list.append(d["feats"].astype(np.float32, copy=False))
        T = d["t_last"].size
        eps_arr_list.append(np.full(T, ep, dtype=np.int64))
        t_last_list.append(d["t_last"])
    feats = np.concatenate(feats_list, axis=0)        # [N, 24, D]
    eps_arr = np.concatenate(eps_arr_list, axis=0)
    t_last = np.concatenate(t_last_list, axis=0)
    tgt = load_targets(task)
    keys = list(zip(tgt["episode_id"].tolist(), tgt["t_last"].tolist()))
    lut = {k: i for i, k in enumerate(keys)}
    rows = np.array([lut[(int(g), int(t))] for g, t in zip(eps_arr, t_last)], dtype=np.int64)
    targets_dict = {tk: tgt[tk][rows] for tk in TARGETS if tk in tgt}
    return feats, eps_arr, targets_dict


def transfer_r2(X_src: np.ndarray, y_src: np.ndarray, X_tgt: np.ndarray, y_tgt: np.ndarray) -> float:
    """Train Ridge on src, test on tgt. Standardize using src statistics
    (since tgt features are not seen at training time)."""
    if y_src.ndim == 1:
        ok_s = np.isfinite(y_src); ok_t = np.isfinite(y_tgt)
    else:
        ok_s = np.isfinite(y_src).all(axis=1); ok_t = np.isfinite(y_tgt).all(axis=1)
    if ok_s.sum() < 100 or ok_t.sum() < 50:
        return float("nan")
    Xs, ys = X_src[ok_s], y_src[ok_s]
    Xt, yt = X_tgt[ok_t], y_tgt[ok_t]
    mu_x = Xs.mean(0); sd_x = Xs.std(0) + 1e-9
    Xs_n = (Xs - mu_x) / sd_x
    Xt_n = (Xt - mu_x) / sd_x   # use src's stats for tgt (transfer-style)
    if ys.ndim == 1:
        mu_y, sd_y = ys.mean(), ys.std() + 1e-9
    else:
        mu_y, sd_y = ys.mean(0), ys.std(0) + 1e-9
    ys_n = (ys - mu_y) / sd_y
    m = Ridge(alpha=1.0); m.fit(Xs_n, ys_n)
    pred = m.predict(Xt_n) * sd_y + mu_y
    return r2_score(yt, pred, multioutput="variance_weighted") if yt.ndim > 1 else r2_score(yt, pred)


def within_task_r2(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    """5-fold GroupKFold Ridge mean R²."""
    if y.ndim == 1:
        ok = np.isfinite(y)
    else:
        ok = np.isfinite(y).all(axis=1)
    if ok.sum() < 100:
        return float("nan")
    X = X[ok]; y = y[ok]; groups = groups[ok]
    gkf = GroupKFold(n_splits=5)
    r2s = []
    for tr, te in gkf.split(X, y, groups=groups):
        Xtr, Xte = X[tr], X[te]; ytr, yte = y[tr], y[te]
        mu_x = Xtr.mean(0); sd_x = Xtr.std(0) + 1e-9
        if ytr.ndim == 1:
            mu_y, sd_y = ytr.mean(), ytr.std() + 1e-9
        else:
            mu_y, sd_y = ytr.mean(0), ytr.std(0) + 1e-9
        Xtr_n = (Xtr - mu_x) / sd_x
        Xte_n = (Xte - mu_x) / sd_x
        ytr_n = (ytr - mu_y) / sd_y
        m = Ridge(alpha=1.0); m.fit(Xtr_n, ytr_n)
        pred = m.predict(Xte_n) * sd_y + mu_y
        r2s.append(r2_score(yte, pred, multioutput="variance_weighted") if yte.ndim > 1 else r2_score(yte, pred))
    return float(np.mean(r2s))


def main():
    # Cache per-task data
    print("[16_cross_task] loading per-task arrays...", flush=True)
    cache = {task: load_task_arrays(task) for task in TASKS}

    rows = []
    for src, tgt_t in PAIRS:
        Xs_full, eps_s, ys_dict = cache[src]
        Xt_full, eps_t, yt_dict = cache[tgt_t]
        for tk in TARGETS:
            if tk not in ys_dict or tk not in yt_dict:
                continue
            for L in range(24):
                Xs = Xs_full[:, L, :]
                Xt = Xt_full[:, L, :]
                y_s = ys_dict[tk]
                y_t = yt_dict[tk]
                tr_r2 = transfer_r2(Xs, y_s, Xt, y_t)
                wn_r2 = within_task_r2(Xt, y_t, eps_t)
                gap = (wn_r2 - tr_r2) if (np.isfinite(wn_r2) and np.isfinite(tr_r2)) else float("nan")
                rows.append({"src_task": src, "tgt_task": tgt_t, "layer": L, "target": tk,
                             "transfer_r2": tr_r2, "within_tgt_r2": wn_r2, "gap": gap})
            print(f"[16_cross_task] {src} → {tgt_t} / {tk} done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(STATS / "cross_task_transfer.csv", index=False)
    print(f"[16_cross_task] wrote stats ({len(df)} rows)", flush=True)

    # Plot: layer × pair gap heatmap per target
    for tk in TARGETS:
        sub = df[df.target == tk]
        if sub.empty:
            continue
        pair_labels = [f"{s}→{t}" for s, t in PAIRS]
        M = np.full((len(PAIRS), 24), np.nan)
        for i, (s, t) in enumerate(PAIRS):
            ss = sub[(sub.src_task == s) & (sub.tgt_task == t)].sort_values("layer")
            for _, r in ss.iterrows():
                M[i, int(r.layer)] = r.gap
        fig, ax = plt.subplots(figsize=(11, 4))
        im = ax.imshow(M, aspect="auto", cmap="coolwarm", origin="lower",
                       extent=[-0.5, 23.5, -0.5, len(PAIRS) - 0.5])
        ax.set_yticks(range(len(PAIRS))); ax.set_yticklabels(pair_labels)
        ax.set_xlabel("Layer"); ax.set_title(f"Cross-task transfer GAP (within−transfer R²) — {tk}")
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        fig.tight_layout()
        fig.savefig(PLOTS / f"cross_task_transfer_gap_{tk}.png", dpi=130)
        plt.close(fig)

    # Also plot transfer R² itself
    for tk in TARGETS:
        sub = df[df.target == tk]
        if sub.empty: continue
        fig, ax = plt.subplots(figsize=(10, 5))
        for s, t in PAIRS:
            ss = sub[(sub.src_task == s) & (sub.tgt_task == t)].sort_values("layer")
            ax.plot(ss.layer, ss.transfer_r2, "-o", markersize=3, label=f"{s}→{t}")
        ax.set_xlabel("Layer"); ax.set_ylabel("Transfer R² (test on tgt)")
        ax.set_title(f"Cross-task transfer R² — {tk}")
        ax.set_xlim(-0.5, 23.5); ax.axhline(0, ls=":", color="gray")
        ax.axvspan(6, 18, alpha=0.05, color="green")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(PLOTS / f"cross_task_transfer_r2_{tk}.png", dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    main()
