#!/usr/bin/env python3
"""Per-task R² vs Layer plots for paradigm v2 functional targets, V-JEPA Variant A.

One PDF + PNG per task with key functional targets as separate curves.
"""
from __future__ import annotations
import os, glob
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

RES = Path("/home/solee/physrepa_tasks/probe/results")
OUT = RES / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Targets to show per task (only those relevant for the task)
COMMON_TARGETS = [
    "ee_velocity__mean",
    "ee_velocity__peak_time_frac",
    "ee_acceleration__peak_to_peak",
    "obj_velocity__mean",
    "obj_acceleration__peak_to_peak",
    "contact_flag__positive_fraction",
    "contact_force_log1p_mag__integral",
]
TASK_EXTRA = {
    "drawer":     ["drawer_joint_pos__mean", "drawer_opening_extent__mean"],
    "peg_insert": ["insertion_depth__mean", "peg_hole_lateral_error__mean"],
    "nut_thread": ["axial_progress__mean", "nut_bolt_relative_angle__mean"],
    "reach":      ["ee_to_target_distance__mean"],
}

VARIANT = "variant_A"


def load_target(task, target):
    p = RES / task / VARIANT / f"{target}.csv"
    if not p.exists(): return None
    df = pd.read_csv(p)
    if "layer" not in df.columns: return None
    mu = df.groupby("layer")["r2"].mean()
    sd = df.groupby("layer")["r2"].std()
    return mu, sd


def plot_task(task):
    targets = COMMON_TARGETS + TASK_EXTRA.get(task, [])
    fig, ax = plt.subplots(figsize=(7, 5))
    for tgt in targets:
        out = load_target(task, tgt)
        if out is None: continue
        mu, sd = out
        layers = mu.index.values
        ax.plot(layers, mu.values, marker='.', label=tgt, alpha=0.85, linewidth=1.2)
        ax.fill_between(layers, mu - sd, mu + sd, alpha=0.10)
    ax.set_xlabel("V-JEPA-L layer")
    ax.set_ylabel("Functional probe $R^2$")
    ax.set_title(f"{task} — Variant A functional probe per-target $R^2$ across 24 layers")
    ax.set_xlim(-0.5, 23.5)
    ax.set_xticks(range(0, 24, 2))
    ax.axhline(0, color='k', lw=0.3, alpha=0.4)
    ax.grid(alpha=0.25)
    ax.legend(loc='lower right', fontsize=7, framealpha=0.85)
    fig.tight_layout()
    pdf = OUT / f"layer_vs_r2_functional_{task}.pdf"
    png = OUT / f"layer_vs_r2_functional_{task}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=110)
    plt.close(fig)
    print(f"wrote {pdf.name}, {png.name}")


def main():
    for task in ["push", "strike", "drawer", "reach", "peg_insert", "nut_thread"]:
        plot_task(task)


if __name__ == "__main__":
    main()
