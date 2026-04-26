#!/usr/bin/env python3
"""Plot layer-vs-R² curves for each task and a cross-task overlay.

Outputs:
  results/plots/<task>_layer_vs_r2.png    — per-task, all 12 (or 6) targets as lines
  results/plots/cross_task_<target>.png   — one figure per headline target with
                                            6 task lines overlaid
  results/plots/heatmap_r2.png            — task × target × layer R² heatmap (compact)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RESULTS = Path("/home/solee/physrepa_tasks/probe/results")
PLOTS = RESULTS / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

ALL_TASKS = ["push", "strike", "reach", "drawer", "peg_insert", "nut_thread"]

# Color/marker scheme
EE_TARGETS = ["ee_position", "ee_velocity", "ee_speed", "ee_direction", "ee_acceleration", "ee_accel_mag"]
OBJ_TARGETS = ["obj_position", "obj_velocity", "obj_speed", "obj_direction", "obj_acceleration", "obj_accel_mag"]
TARGET_COLORS = {
    "ee_position": "#1f77b4",  "obj_position": "#1f77b4",
    "ee_velocity": "#ff7f0e",  "obj_velocity": "#ff7f0e",
    "ee_speed":    "#2ca02c",  "obj_speed":    "#2ca02c",
    "ee_direction":"#d62728",  "obj_direction":"#d62728",
    "ee_acceleration": "#9467bd",  "obj_acceleration": "#9467bd",
    "ee_accel_mag":"#8c564b",  "obj_accel_mag":"#8c564b",
}

TASK_COLORS = {
    "push": "#1f77b4", "strike": "#ff7f0e", "reach": "#2ca02c",
    "drawer": "#d62728", "peg_insert": "#9467bd", "nut_thread": "#8c564b",
}


def _summary_df(task: str) -> pd.DataFrame | None:
    p = RESULTS / task / "variant_A" / "_summary.csv"
    return pd.read_csv(p) if p.exists() else None


def plot_per_task(task: str) -> Path:
    df = _summary_df(task)
    if df is None:
        return None
    targets = list(df["target"].unique())
    ee = [t for t in targets if t.startswith("ee_")]
    obj = [t for t in targets if t.startswith("obj_")]
    has_obj = bool(obj)

    fig, axes = plt.subplots(1, 2 if has_obj else 1, figsize=(13 if has_obj else 8, 5), sharey=True, squeeze=False)
    ax_ee = axes[0, 0]
    ax_obj = axes[0, 1] if has_obj else None

    for tgt in ee:
        sub = df[df["target"] == tgt].sort_values("layer")
        ax_ee.plot(sub["layer"], sub["r2_mean"], "-o", color=TARGET_COLORS.get(tgt, "k"),
                   label=tgt.replace("ee_", ""), markersize=4, linewidth=1.5)
        ax_ee.fill_between(sub["layer"], sub["r2_mean"] - sub["r2_std"],
                           sub["r2_mean"] + sub["r2_std"], color=TARGET_COLORS.get(tgt, "k"), alpha=0.10)
    ax_ee.set_xlabel("Layer")
    ax_ee.set_ylabel("R² (variance-weighted, 5-fold mean)")
    ax_ee.set_title(f"{task}  EE targets")
    ax_ee.set_xlim(-0.5, 23.5)
    ax_ee.set_ylim(-0.05, 1.02)
    ax_ee.axvspan(6, 18, alpha=0.05, color="green", label="_PEZ band")
    ax_ee.axhline(0.5, ls=":", color="gray", lw=0.8)
    ax_ee.grid(alpha=0.3)
    ax_ee.legend(loc="lower right", fontsize=8)

    if ax_obj is not None:
        for tgt in obj:
            sub = df[df["target"] == tgt].sort_values("layer")
            ax_obj.plot(sub["layer"], sub["r2_mean"], "-o", color=TARGET_COLORS.get(tgt, "k"),
                        label=tgt.replace("obj_", ""), markersize=4, linewidth=1.5)
            ax_obj.fill_between(sub["layer"], sub["r2_mean"] - sub["r2_std"],
                                sub["r2_mean"] + sub["r2_std"], color=TARGET_COLORS.get(tgt, "k"), alpha=0.10)
        ax_obj.set_xlabel("Layer")
        ax_obj.set_title(f"{task}  Object targets")
        ax_obj.set_xlim(-0.5, 23.5)
        ax_obj.axvspan(6, 18, alpha=0.05, color="green")
        ax_obj.axhline(0.5, ls=":", color="gray", lw=0.8)
        ax_obj.grid(alpha=0.3)
        ax_obj.legend(loc="lower right", fontsize=8)

    fig.suptitle(f"V-JEPA 2 ViT-L Variant A — {task}  (shaded = ±1 std across 5 folds; green band = spec § 16 PEZ range L6–18)",
                 fontsize=10)
    fig.tight_layout()
    out = PLOTS / f"{task}_layer_vs_r2.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_cross_task(target: str) -> Path | None:
    fig, ax = plt.subplots(figsize=(9, 5))
    drew_any = False
    for task in ALL_TASKS:
        df = _summary_df(task)
        if df is None:
            continue
        sub = df[df["target"] == target].sort_values("layer")
        if sub.empty:
            continue
        ax.plot(sub["layer"], sub["r2_mean"], "-o", color=TASK_COLORS[task],
                label=task, markersize=4, linewidth=1.5)
        ax.fill_between(sub["layer"], sub["r2_mean"] - sub["r2_std"],
                        sub["r2_mean"] + sub["r2_std"], color=TASK_COLORS[task], alpha=0.12)
        drew_any = True
    if not drew_any:
        plt.close(fig)
        return None
    ax.set_xlabel("Layer")
    ax.set_ylabel("R² (variance-weighted, 5-fold mean)")
    ax.set_title(f"Cross-task layer-vs-R² — {target}")
    ax.set_xlim(-0.5, 23.5)
    ax.set_ylim(-0.05, 1.02)
    ax.axvspan(6, 18, alpha=0.05, color="green")
    ax.axhline(0.5, ls=":", color="gray", lw=0.8)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    out = PLOTS / f"cross_task_{target}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_heatmap() -> Path:
    """Compact: rows = (task, target), cols = layer, color = R²."""
    rows = []
    labels = []
    for task in ALL_TASKS:
        df = _summary_df(task)
        if df is None:
            continue
        for tgt in df["target"].unique():
            sub = df[df["target"] == tgt].sort_values("layer")
            if len(sub) != 24:
                continue
            rows.append(sub["r2_mean"].to_numpy())
            labels.append(f"{task}/{tgt}")
    arr = np.stack(rows)
    n = len(labels)
    fig, ax = plt.subplots(figsize=(11, max(6, 0.22 * n)))
    im = ax.imshow(arr, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(24))
    ax.set_xlabel("Layer")
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title("R² heatmap — V-JEPA 2 ViT-L Variant A across all (task, target) × 24 layers")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("R² mean (5-fold)")
    fig.tight_layout()
    out = PLOTS / "heatmap_r2.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main():
    written = []
    for task in ALL_TASKS:
        p = plot_per_task(task)
        if p:
            written.append(p)
    for tgt in ["ee_position", "ee_velocity", "ee_speed", "ee_direction", "ee_acceleration", "ee_accel_mag",
                "obj_position", "obj_velocity", "obj_speed", "obj_direction", "obj_acceleration", "obj_accel_mag"]:
        p = plot_cross_task(tgt)
        if p:
            written.append(p)
    written.append(plot_heatmap())
    print(f"wrote {len(written)} plots to {PLOTS}")
    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()
