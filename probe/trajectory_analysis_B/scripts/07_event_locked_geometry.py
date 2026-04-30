#!/usr/bin/env python3
"""Event-locked latent geometry around contact onset (push/strike/drawer).

Per Codex spec:
- Detect first contact onset per episode: t0 = min{t : contact[t-1]=0 and contact[t]=1}
- Use relative window τ ∈ [-H, H] with H=8
- Per layer per τ compute:
    speed_t       = ||z_{t+1} - z_t||
    curvature_t   = arccos(<u_t, u_{t+1}> / (||u_t|| * ||u_{t+1}|| + ε))
    tortuosity_t  = (Σ_{k=t-w}^{t+w-1} ||u_k||) / (||z_{t+w} - z_{t-w}|| + ε)
    PR_t          = (Σλ)² / Σλ²  on cov of {z_{t-w}..z_{t+w}}, w=4
- Aggregate mean/SEM across episodes per (task, layer, τ, metric)

For drawer (no clean contact_flag), use velocity-magnitude crossing of a threshold
as proxy event ('motion onset').

Outputs:
- results/stats/event_locked_<task>.csv (long format: task, layer, tau, metric, mean, sem)
- results/plots/event_locked_<task>.png (heatmaps: layer × tau, separate panels for speed, curvature, PR)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROBE_ROOT = Path("/home/solee/physrepa_tasks/probe")
sys.path.insert(0, str(PROBE_ROOT))

from ta_utils.loader import all_trajectories
from utils.io import load_targets


PLOTS = ROOT / "results" / "plots"
STATS = ROOT / "results" / "stats"
PLOTS.mkdir(parents=True, exist_ok=True)
STATS.mkdir(parents=True, exist_ok=True)

H = 8
W_LOCAL = 4   # half-window for tortuosity / PR


def detect_onset_contact(contact_flag: np.ndarray) -> int | None:
    """First t where contact_flag[t-1]==0 and contact_flag[t]==1."""
    cf = contact_flag.astype(np.int32).reshape(-1)
    edges = np.where((cf[1:] == 1) & (cf[:-1] == 0))[0]
    if edges.size == 0:
        return None
    return int(edges[0]) + 1


def detect_onset_sustained(speed_seq: np.ndarray, frac: float = 0.3, k: int = 3) -> int | None:
    """Drawer / motion-onset proxy: first t where speed > frac*max for k
    consecutive windows. Per Codex spec (sustained criterion)."""
    if speed_seq.size < k:
        return None
    thr = frac * float(speed_seq.max())
    above = (speed_seq > thr).astype(np.int32)
    # Find first run of k consecutive 1s
    csum = np.convolve(above, np.ones(k, dtype=np.int32), mode="valid")
    hits = np.where(csum == k)[0]
    if hits.size == 0:
        return None
    return int(hits[0])


def event_locked_metrics(Z: np.ndarray, t0: int, H: int = 8, w_local: int = 4) -> dict:
    """Z: [T, D]. Returns per-tau metrics around t0; tau ∈ [-H, H]."""
    T = Z.shape[0]
    out: dict[str, np.ndarray] = {
        "speed": np.full(2 * H + 1, np.nan),
        "curvature": np.full(2 * H + 1, np.nan),
        "tortuosity": np.full(2 * H + 1, np.nan),
        "pr": np.full(2 * H + 1, np.nan),
    }
    # Pre-compute step vectors and speeds
    steps = np.diff(Z, axis=0)                          # [T-1, D]
    step_norms = np.linalg.norm(steps, axis=1)          # [T-1]
    for i, tau in enumerate(range(-H, H + 1)):
        t = t0 + tau
        if t < 0 or t >= T - 1:
            continue
        # Speed at t
        out["speed"][i] = float(step_norms[t])
        # Curvature: angle between u_t and u_{t+1}
        if t + 1 < T - 1:
            num = float((steps[t] * steps[t + 1]).sum())
            den = float(step_norms[t] * step_norms[t + 1] + 1e-12)
            cos = max(-1.0, min(1.0, num / den))
            out["curvature"][i] = float(np.arccos(cos))
        # Tortuosity over [t-w, t+w]
        ts0 = max(0, t - w_local)
        te = min(T - 1, t + w_local)
        if te > ts0:
            path = float(step_norms[ts0:te].sum())
            direct = float(np.linalg.norm(Z[te] - Z[ts0]) + 1e-12)
            out["tortuosity"][i] = path / direct
        # Participation ratio over local window
        if te - ts0 + 1 >= 2:
            local = Z[ts0:te + 1]                       # [<=2w+1, D]
            local_c = local - local.mean(axis=0, keepdims=True)
            # Use eigvals of small covariance: cov is D×D but rank limited by samples
            # Compute via Gram trick: eigvals of (X X^T) = eigvals of (X^T X) (nonzero ones)
            G = local_c @ local_c.T
            eigvals = np.linalg.eigvalsh(G)
            eigvals = np.clip(eigvals, 0, None)
            s = eigvals.sum()
            s2 = (eigvals * eigvals).sum()
            if s2 > 0:
                out["pr"][i] = float((s * s) / s2)
    return out


def main():
    rows = []
    for task in ["push", "strike", "drawer"]:
        print(f"[07_event] {task} loading ...", flush=True)
        trajs = all_trajectories(task)
        tgt = load_targets(task)
        # contact_flag is per-window in our windowed targets (NOT in tgt — that's just for kinematics).
        # We need to read contact info from per-episode parquets directly.
        # Simpler: for push/strike, contact_flag and full trajectory exist in parquets.
        # We'll use saved targets' "ee_speed" or finite-diff velocity to detect for drawer.

        # Load per-episode contact_flag from parquets directly
        import pyarrow.parquet as pq
        from utils.dataset import parquet_for_episode

        layer_metric_arr = {}   # (layer, metric) -> list of [2H+1] arrays
        for L in range(24):
            for m in ("speed", "curvature", "tortuosity", "pr"):
                layer_metric_arr[(L, m)] = []

        kept_episodes = 0
        for tj in trajs:
            ep = int(tj["episode_id"])
            try:
                df = pq.read_table(parquet_for_episode(task, ep)).to_pandas()
            except Exception:
                continue
            t_last = tj["t_last"]
            # Build contact sequence aligned to windows: contact_flag at frame t (window's last frame).
            cf_col = "physics_gt.contact_flag"
            if cf_col not in df.columns:
                continue
            cf_full = np.array([float(v) if not isinstance(v, np.ndarray) else float(v.flatten()[0])
                                for v in df[cf_col].tolist()], dtype=np.float32)
            # cf_full is per-frame. Map to per-window via t_last indexing.
            cf_windowed = cf_full[t_last]
            # Detect onset
            if task == "drawer":
                # Sustained motion-onset proxy on handle/object speed (per Codex).
                idx_t = list(zip(tgt["episode_id"].tolist(), tgt["t_last"].tolist()))
                lut = {k: i for i, k in enumerate(idx_t)}
                rows_idx = np.array([lut[(ep, int(t))] for t in t_last], dtype=np.int64)
                obj_speed = tgt.get("obj_speed")
                if obj_speed is None:
                    speed_for_onset = tgt["ee_speed"][rows_idx]
                else:
                    speed_for_onset = obj_speed[rows_idx]
                t0 = detect_onset_sustained(speed_for_onset, frac=0.30, k=3)
            else:
                t0 = detect_onset_contact(cf_windowed)
            if t0 is None:
                continue
            kept_episodes += 1

            feats = tj["feats"]   # [T, 24, D]
            for L in range(24):
                Z = feats[:, L, :]
                m = event_locked_metrics(Z, t0, H=H, w_local=W_LOCAL)
                for metric_name in ("speed", "curvature", "tortuosity", "pr"):
                    layer_metric_arr[(L, metric_name)].append(m[metric_name])

        print(f"[07_event] {task}: kept {kept_episodes}/{len(trajs)} eps with detectable event", flush=True)

        # Aggregate
        for L in range(24):
            for metric_name in ("speed", "curvature", "tortuosity", "pr"):
                arrs = np.stack(layer_metric_arr[(L, metric_name)], axis=0) if layer_metric_arr[(L, metric_name)] else np.zeros((0, 2 * H + 1))
                for i, tau in enumerate(range(-H, H + 1)):
                    col = arrs[:, i] if arrs.size > 0 else np.array([])
                    col = col[np.isfinite(col)]
                    if col.size == 0:
                        continue
                    rows.append({
                        "task": task, "layer": L, "tau": tau,
                        "metric": metric_name,
                        "mean": float(col.mean()),
                        "sem": float(col.std() / max(np.sqrt(col.size), 1)),
                        "n": int(col.size),
                    })

    df = pd.DataFrame(rows)
    df.to_csv(STATS / "event_locked.csv", index=False)
    print(f"[07_event] wrote stats ({len(df)} rows)", flush=True)

    # Plotting per task: 4 metrics × 24 layers × (2H+1) tau heatmaps
    for task in df.task.unique():
        sub = df[df.task == task]
        fig, axes = plt.subplots(2, 2, figsize=(15, 9), squeeze=False)
        for ax, metric in zip(axes.flatten(), ["speed", "curvature", "tortuosity", "pr"]):
            mat = np.full((24, 2 * H + 1), np.nan)
            for _, r in sub[sub.metric == metric].iterrows():
                mat[int(r.layer), int(r.tau) + H] = r["mean"]
            im = ax.imshow(mat, aspect="auto", cmap="viridis", origin="lower",
                           extent=[-H - 0.5, H + 0.5, -0.5, 23.5])
            ax.axvline(0, color="white", linestyle="--", linewidth=0.8)
            ax.set_xlabel("τ (windows from contact onset)")
            ax.set_ylabel("Layer")
            ax.set_title(f"{metric} — {task}")
            fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        fig.suptitle(f"Event-locked latent geometry around contact onset — {task}", fontsize=11)
        fig.tight_layout()
        fig.savefig(PLOTS / f"event_locked_{task}.png", dpi=120)
        plt.close(fig)
        print(f"[07_event] wrote event_locked_{task}.png", flush=True)


if __name__ == "__main__":
    main()
