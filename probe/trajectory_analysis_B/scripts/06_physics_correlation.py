#!/usr/bin/env python3
"""Physics-conditioning analysis: do layer-wise embeddings encode physics
parameters (mass, friction, damping)?

Per task we have per-episode physics_gt fields stored in the parquet metadata.
For each layer, we test whether the *episode-mean embedding* correlates with
each physics parameter via:
  - (a) episode-mean L2 distance: |<emb>(ep1) - <emb>(ep2)|  vs |Δphysics|
        → Spearman rho summarized as "metric correspondence"
  - (b) linear regression: physics ~ W · episode_mean_emb  → R² per (layer, param)

Outputs:
- results/stats/physics_corr.csv: task, layer, param, metric_corr, regression_r2
- results/plots/physics_emb_per_layer_<task>.png: layer (x) vs regression R² (y),
  one line per physics param
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ta_utils.loader import ALL_TASKS, all_trajectories
from utils.io import load_common  # type: ignore


PROBE_ROOT = Path("/home/solee/physrepa_tasks/probe")
sys.path.insert(0, str(PROBE_ROOT))

PLOTS = ROOT / "results" / "plots"
STATS = ROOT / "results" / "stats"
PLOTS.mkdir(parents=True, exist_ok=True)
STATS.mkdir(parents=True, exist_ok=True)


def physics_param_columns(task: str) -> list[str]:
    """Per-task list of expected physics_gt parameter columns. Best-effort:
    we will only use columns actually present in the parquet."""
    candidates = {
        "push": ["physics_gt.mass", "physics_gt.obj_friction", "physics_gt.surface_friction"],
        "strike": ["physics_gt.mass", "physics_gt.friction", "physics_gt.surface_friction", "physics_gt.restitution"],
        "drawer": ["physics_gt.drawer_joint_damping"],
        "peg_insert": ["physics_gt.held_friction", "physics_gt.fixed_friction", "physics_gt.held_mass"],
        "nut_thread": ["physics_gt.held_friction", "physics_gt.fixed_friction", "physics_gt.held_mass"],
        "reach": [],
    }
    return candidates.get(task, [])


def load_episode_physics(task: str, episode_id: int) -> dict:
    """Read the FIRST row of episode parquet — physics params are constant per ep."""
    common = load_common()
    base = Path(common["dataset_root"]) / task / "data"
    for chunk in sorted(base.glob("chunk-*")):
        p = chunk / f"episode_{episode_id:06d}.parquet"
        if p.exists():
            tbl = pq.read_table(p, columns=None)
            df = tbl.slice(0, 1).to_pandas()
            return {c: float(df[c].iloc[0]) for c in df.columns
                    if c.startswith("physics_gt.") and df[c].dtype.kind in "fi"}
    return {}


def main():
    rows = []
    for task in ALL_TASKS:
        params = physics_param_columns(task)
        if not params:
            print(f"[06_physics] {task}: no physics params declared; skipping", flush=True)
            continue
        print(f"[06_physics] {task} loading ...", flush=True)
        trajs = all_trajectories(task)
        ep_physics = []
        for t in trajs:
            ep_physics.append(load_episode_physics(task, int(t["episode_id"])))

        # Fixed param list = intersection of present
        present = set.intersection(*(set(p.keys()) for p in ep_physics if p))
        params_use = [p for p in params if p in present]
        if not params_use:
            print(f"[06_physics] {task}: declared params not present in parquet — taking all numeric physics_gt: {sorted(present)[:6]}...", flush=True)
            params_use = sorted(present)[:6]
        print(f"[06_physics] {task}: using params {params_use}", flush=True)

        # Build [n_eps, n_params] target matrix
        Y = np.array([[ep_physics[i].get(p, np.nan) for p in params_use]
                      for i in range(len(trajs))], dtype=np.float32)
        ok = np.isfinite(Y).all(axis=1)
        if ok.sum() < 5:
            print(f"[06_physics] {task}: not enough valid eps — skipping", flush=True)
            continue
        Y = Y[ok]
        trajs_ok = [t for i, t in enumerate(trajs) if ok[i]]

        for L in range(24):
            E = np.stack([t["feats"][:, L, :].mean(axis=0) for t in trajs_ok], axis=0)  # [n_eps, 1024]
            for j, param in enumerate(params_use):
                y = Y[:, j]
                # Standardize and Ridge with leave-one-out-ish split (just train_test 80/20)
                rng = np.random.default_rng(42)
                idx = rng.permutation(E.shape[0])
                ntr = max(int(E.shape[0] * 0.8), 4)
                tr, te = idx[:ntr], idx[ntr:]
                if te.size == 0:
                    continue
                Em, Es = E[tr].mean(0), E[tr].std(0) + 1e-9
                Ym, Ys = y[tr].mean(), y[tr].std() + 1e-9
                Etr = (E[tr] - Em) / Es
                Ete = (E[te] - Em) / Es
                ytr = (y[tr] - Ym) / Ys
                yte_raw = y[te]
                # Ridge
                model = Ridge(alpha=1.0)
                model.fit(Etr, ytr)
                pred = model.predict(Ete) * Ys + Ym
                ss_res = float(((yte_raw - pred) ** 2).sum())
                ss_tot = float(((yte_raw - yte_raw.mean()) ** 2).sum() + 1e-9)
                r2 = 1 - ss_res / ss_tot

                # Metric correspondence: pairwise distances
                if E.shape[0] >= 4:
                    D_e = np.linalg.norm(E[:, None, :] - E[None, :, :], axis=2)
                    D_p = np.abs(y[:, None] - y[None, :])
                    iu = np.triu_indices_from(D_e, k=1)
                    rho, _ = spearmanr(D_e[iu], D_p[iu])
                else:
                    rho = float("nan")

                rows.append({"task": task, "layer": L, "param": param,
                             "regression_r2": float(r2),
                             "metric_corr_spearman": float(rho)})

        print(f"[06_physics] {task}: done ({len(params_use)} params × 24 layers)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(STATS / "physics_corr.csv", index=False)
    print(f"[06_physics] wrote {STATS / 'physics_corr.csv'} ({len(df)} rows)", flush=True)

    # Per-task plot
    for task in df.task.unique():
        sub = df[df.task == task]
        params = sub.param.unique()
        fig, ax = plt.subplots(figsize=(9, 5))
        for p in params:
            s = sub[sub.param == p].sort_values("layer")
            ax.plot(s.layer, s.regression_r2, "-o", label=p.replace("physics_gt.", ""), markersize=4)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Ridge R² (episode-mean emb → physics param)")
        ax.set_title(f"Physics encoding per layer — {task}")
        ax.set_xlim(-0.5, 23.5)
        ax.axhline(0, ls=":", color="gray")
        ax.axvspan(6, 18, alpha=0.05, color="green")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(PLOTS / f"physics_emb_per_layer_{task}.png", dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    main()
