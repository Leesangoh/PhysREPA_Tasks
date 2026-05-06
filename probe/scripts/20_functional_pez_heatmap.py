#!/usr/bin/env python3
"""J — 6 task × 5 model functional-PEZ heatmap (paper main figure candidate).

For each (task, model_variant) cell, compute peak R² (best layer) for a
selected target family. Produces:

  probe/results/figures/functional_pez_heatmap_<family>.pdf
  probe/results/figures/functional_pez_heatmap_summary.csv

Models considered:
  V-JEPA A    = variant_A
  V-JEPA B    = variant_B
  VideoMAE A  = variant_A_videomae_large
  VideoMAE B  = variant_B_videomae_large
  DINOv2 A    = variant_A_dinov2_large

Target families (Codex collapsed redundancy):
  magnitude   = ee_velocity__mean / ee_acceleration__peak_to_peak / obj_velocity__mean
  contact     = contact_flag__positive_fraction / contact_force_log1p_mag__integral
  timing      = ee_velocity__peak_time_frac / contact_flag__first_event_time_frac
  progress    = drawer_joint_pos__mean / insertion_depth__total_variation / etc

Incrementally update — running this anytime gives the current state of the
matrix (missing cells reported as NaN in the heatmap).

Usage:
  /isaac-sim/python.sh probe/scripts/20_functional_pez_heatmap.py [--target T] [--family F]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROBE_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROBE_ROOT / "results"
FIG_DIR = RESULTS / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

TASKS = ["push", "strike", "drawer", "reach", "peg_insert", "nut_thread"]

# (variant_dir_name, display_label)
MODELS = [
    ("variant_A",                  "V-JEPA A"),
    ("variant_B",                  "V-JEPA B"),
    ("variant_A_videomae_large",   "VideoMAE A"),
    ("variant_B_videomae_large",   "VideoMAE B"),
    ("variant_A_dinov2_large",     "DINOv2 A"),
]

# Headline single-target picks per family (one canonical functional per family,
# avoiding mean/integral redundancy)
FAMILY_TARGETS = {
    "magnitude_ee_velocity":     "ee_velocity__mean",
    "magnitude_ee_acceleration": "ee_acceleration__peak_to_peak",
    "magnitude_obj_velocity":    "obj_velocity__mean",
    "contact_occupancy":         "contact_flag__positive_fraction",
    "contact_force_integral":    "contact_force_log1p_mag__integral",
    "timing_velocity":           "ee_velocity__peak_time_frac",
    "timing_first_event":        "contact_flag__first_event_time_frac",
}


def load_peak_r2(task: str, variant: str, target: str):
    """Return (peak_r2, peak_layer) or (NaN, NaN) if missing."""
    p = RESULTS / task / variant / f"{target}.csv"
    if not p.exists():
        return float("nan"), -1
    try:
        df = pd.read_csv(p)
        per = df.groupby("layer")["r2"].mean()
        return float(per.max()), int(per.idxmax())
    except Exception:
        return float("nan"), -1


def build_matrix(target: str):
    """Return 6x5 peak R² matrix + 6x5 peak-layer matrix."""
    R = np.full((len(TASKS), len(MODELS)), np.nan)
    L = np.full((len(TASKS), len(MODELS)), -1, dtype=int)
    for i, task in enumerate(TASKS):
        for j, (var, _label) in enumerate(MODELS):
            r2, layer = load_peak_r2(task, var, target)
            R[i, j] = r2
            L[i, j] = layer
    return R, L


def plot_heatmap(R, L, target: str, out_pdf: Path):
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    cmap = plt.get_cmap("viridis")
    im = ax.imshow(R, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([m[1] for m in MODELS], rotation=20, ha="right")
    ax.set_yticks(range(len(TASKS)))
    ax.set_yticklabels(TASKS)
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            v = R[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}\n@L{L[i, j]:02d}",
                        ha="center", va="center",
                        color="white" if v < 0.65 else "black", fontsize=8)
            else:
                ax.text(j, i, "—", ha="center", va="center", color="grey", fontsize=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("peak $R^2$", rotation=270, labelpad=15)
    ax.set_title(f"Functional-PEZ peak R² (target: {target})")
    fig.tight_layout()
    fig.savefig(out_pdf, format="pdf")
    plt.close(fig)


def write_summary_csv(out_csv: Path):
    rows = []
    for fam, tgt in FAMILY_TARGETS.items():
        R, L = build_matrix(tgt)
        for i, task in enumerate(TASKS):
            for j, (var, label) in enumerate(MODELS):
                rows.append({
                    "family": fam,
                    "target": tgt,
                    "task": task,
                    "variant": var,
                    "model_label": label,
                    "peak_r2": R[i, j],
                    "peak_layer": L[i, j],
                })
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", default=None,
                   help="single target (e.g. ee_acceleration__peak_to_peak); default = all family targets")
    args = p.parse_args()

    targets = [args.target] if args.target else list(FAMILY_TARGETS.values())
    summary_csv = FIG_DIR / "functional_pez_heatmap_summary.csv"
    write_summary_csv(summary_csv)
    print(f"summary CSV: {summary_csv}")

    for tgt in targets:
        R, L = build_matrix(tgt)
        n_done = int(np.sum(np.isfinite(R)))
        n_total = R.size
        out_pdf = FIG_DIR / f"functional_pez_heatmap_{tgt}.pdf"
        plot_heatmap(R, L, tgt, out_pdf)
        print(f"[{tgt}] {n_done}/{n_total} cells → {out_pdf}")


if __name__ == "__main__":
    main()
