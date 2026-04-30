#!/usr/bin/env python3
"""Phase-conditional probing (Codex bonus tier).

Split episode windows by phase OR contact_flag and re-probe per condition.
"Does the representation differ between approach vs contact phases?"

Two conditioning schemes:
  A. contact_flag conditional: split into {contact_flag=0} vs {contact_flag=1}
     - Useful for push/strike/drawer (contact varies)
     - Skip peg/nut (always-contact) and reach (no contact)
  B. phase conditional: split by phase ∈ {0,1,2,...} (approach/contact/withdraw codes)
     - For tasks where phase varies across episodes; reach has phase

For each (task, layer, target, condition_value):
  - Subselect windows matching the condition
  - Run 5-fold GroupKFold Ridge probe
  - Report R² mean / std

Compare to unconditional R² (whole-episode probe) → "localization" of physics
emergence by phase.

Targets: ee_position, ee_velocity, ee_acceleration, contact_force_log1p_mag.
Variant: A (1024-d).

Outputs:
- results/stats/phase_conditional.csv: task, layer, target, condition_type, condition_value, n_windows, r2_mean, r2_std
- results/plots/phase_conditional_<task>_<target>.png: layer (x) × R² (y), one line per condition
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
import pyarrow.parquet as pq
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

PROBE_ROOT = Path("/home/solee/physrepa_tasks/probe")
sys.path.insert(0, str(PROBE_ROOT))
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.io import list_cached_episodes, load_episode_features, load_targets
from utils.dataset import parquet_for_episode


PLOTS = ROOT / "results" / "plots"
STATS = ROOT / "results" / "stats"
PLOTS.mkdir(parents=True, exist_ok=True)
STATS.mkdir(parents=True, exist_ok=True)


N_EPS = 60
TARGETS = ["ee_position", "ee_velocity", "ee_acceleration", "contact_force_log1p_mag"]
TASKS_CONTACT_VARYING = ["push", "strike", "drawer"]   # contact_flag varies
TASKS_PHASE_VARYING = ["push", "strike", "reach", "drawer", "peg_insert", "nut_thread"]


def fold_r2(X, y, groups, n_splits=5):
    if y.ndim == 1:
        ok = np.isfinite(y)
    else:
        ok = np.isfinite(y).all(axis=1)
    if ok.sum() < 100:
        return float("nan"), float("nan")
    X = X[ok]; y = y[ok]; groups = groups[ok]
    n_unique = np.unique(groups).size
    if n_unique < n_splits:
        return float("nan"), float("nan")
    gkf = GroupKFold(n_splits=n_splits)
    r2s = []
    for tr, te in gkf.split(X, y, groups=groups):
        if tr.size < 50 or te.size < 20:
            continue
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
    if not r2s:
        return float("nan"), float("nan")
    return float(np.mean(r2s)), float(np.std(r2s))


def main():
    rows = []
    for task in TASKS_PHASE_VARYING:
        print(f"[17_phase_cond] {task} loading ...", flush=True)
        eps = list_cached_episodes(task, "A")
        rng = np.random.default_rng(42)
        if len(eps) > N_EPS:
            eps = sorted(rng.choice(eps, size=N_EPS, replace=False).tolist())

        feats_list, eps_arr_list, t_last_list = [], [], []
        contact_list, phase_list = [], []
        for ep in eps:
            d = load_episode_features(task, "A", ep)
            feats_list.append(d["feats"].astype(np.float32, copy=False))
            T = d["t_last"].size
            eps_arr_list.append(np.full(T, ep, dtype=np.int64))
            t_last_list.append(d["t_last"])
            # Load contact_flag and phase from parquet for these windows
            try:
                df = pq.read_table(parquet_for_episode(task, ep)).to_pandas()
                if "physics_gt.contact_flag" in df.columns:
                    cf_all = np.array([float(v) if not isinstance(v, np.ndarray) else float(v.flatten()[0])
                                       for v in df["physics_gt.contact_flag"].tolist()], dtype=np.float32)
                    contact_list.append(cf_all[d["t_last"]])
                else:
                    contact_list.append(np.full(T, np.nan, dtype=np.float32))
                if "physics_gt.phase" in df.columns:
                    ph_all = np.array([float(v) if not isinstance(v, np.ndarray) else float(v.flatten()[0])
                                       for v in df["physics_gt.phase"].tolist()], dtype=np.float32)
                    phase_list.append(ph_all[d["t_last"]])
                else:
                    phase_list.append(np.full(T, np.nan, dtype=np.float32))
            except Exception:
                contact_list.append(np.full(T, np.nan, dtype=np.float32))
                phase_list.append(np.full(T, np.nan, dtype=np.float32))

        feats = np.concatenate(feats_list, axis=0)
        eps_arr = np.concatenate(eps_arr_list, axis=0)
        t_last = np.concatenate(t_last_list, axis=0)
        contact = np.concatenate(contact_list, axis=0)
        phase = np.concatenate(phase_list, axis=0)

        tgt = load_targets(task)
        keys_tgt = list(zip(tgt["episode_id"].tolist(), tgt["t_last"].tolist()))
        lut = {k: i for i, k in enumerate(keys_tgt)}
        rows_idx = np.array([lut[(int(g), int(t))] for g, t in zip(eps_arr, t_last)], dtype=np.int64)
        target_data = {tk: tgt[tk][rows_idx] for tk in TARGETS if tk in tgt.files}

        for L in range(24):
            X_layer = feats[:, L, :]
            for tk, y in target_data.items():
                # Unconditional baseline
                r2_m, r2_s = fold_r2(X_layer, y, eps_arr)
                rows.append({"task": task, "layer": L, "target": tk, "condition_type": "all",
                             "condition_value": "all", "n_windows": int(eps_arr.size),
                             "r2_mean": r2_m, "r2_std": r2_s})

                # Contact-conditional (only for contact-varying tasks)
                if task in TASKS_CONTACT_VARYING and not np.isnan(contact).all():
                    for cv in [0.0, 1.0]:
                        mask = (contact == cv)
                        if mask.sum() < 200:
                            continue
                        r2_m, r2_s = fold_r2(X_layer[mask], y[mask], eps_arr[mask])
                        rows.append({"task": task, "layer": L, "target": tk,
                                     "condition_type": "contact", "condition_value": str(int(cv)),
                                     "n_windows": int(mask.sum()),
                                     "r2_mean": r2_m, "r2_std": r2_s})

                # Phase-conditional (top 3 most-frequent phases)
                if not np.isnan(phase).all():
                    uniq, counts = np.unique(phase[np.isfinite(phase)], return_counts=True)
                    top = uniq[np.argsort(-counts)][:3]
                    for pv in top:
                        mask = (phase == pv)
                        if mask.sum() < 200:
                            continue
                        r2_m, r2_s = fold_r2(X_layer[mask], y[mask], eps_arr[mask])
                        rows.append({"task": task, "layer": L, "target": tk,
                                     "condition_type": "phase", "condition_value": f"phase={int(pv)}",
                                     "n_windows": int(mask.sum()),
                                     "r2_mean": r2_m, "r2_std": r2_s})
        print(f"[17_phase_cond] {task}: done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(STATS / "phase_conditional.csv", index=False)
    print(f"[17_phase_cond] wrote stats ({len(df)} rows)", flush=True)

    # Plots: per (task, target) layer × R² per condition
    for task in TASKS_PHASE_VARYING:
        for tk in TARGETS:
            sub = df[(df.task == task) & (df.target == tk)]
            if sub.empty:
                continue
            fig, ax = plt.subplots(figsize=(10, 5))
            for cond_label, cond_df in sub.groupby(["condition_type", "condition_value"]):
                ct, cv = cond_label
                lbl = f"{ct}={cv}"
                cd = cond_df.sort_values("layer")
                ax.plot(cd.layer, cd.r2_mean, "-o", markersize=3, linewidth=1.2, label=lbl)
            ax.set_xlabel("Layer"); ax.set_ylabel("R²")
            ax.set_title(f"Phase/contact-conditional probing — {task} / {tk}")
            ax.set_xlim(-0.5, 23.5); ax.axhline(0, ls=":", color="gray")
            ax.axvspan(6, 18, alpha=0.05, color="green")
            ax.legend(fontsize=7, loc="lower right"); ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(PLOTS / f"phase_conditional_{task}_{tk}.png", dpi=130)
            plt.close(fig)


if __name__ == "__main__":
    main()
