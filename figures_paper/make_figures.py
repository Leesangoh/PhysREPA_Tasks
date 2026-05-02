#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


BASE = Path("/home/solee/physrepa_tasks")
PROBE = BASE / "probe"
OUT = BASE / "figures_paper"
OUT.mkdir(parents=True, exist_ok=True)

TARGET_ORDER = [
    "ee_velocity",
    "ee_acceleration",
    "contact_flag",
    "contact_force_log1p_mag",
]
TARGET_LABELS = {
    "ee_velocity": "EE velocity",
    "ee_acceleration": "EE acceleration",
    "contact_flag": "Contact flag",
    "contact_force_log1p_mag": "Contact force",
}
TARGET_SHORT = {
    "ee_velocity": "vel",
    "ee_acceleration": "acc",
    "contact_flag": "flag",
    "contact_force_log1p_mag": "force",
}
TARGET_COLORS = {
    "ee_velocity": "#377eb8",
    "ee_acceleration": "#ff7f00",
    "contact_flag": "#4daf4a",
    "contact_force_log1p_mag": "#e41a1c",
}
TASK_COLORS = {
    "push": "#377eb8",
    "strike": "#ff7f00",
    "drawer": "#4daf4a",
}


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.2,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_label(ax: plt.Axes, text: str) -> None:
    ax.text(
        0.0,
        1.02,
        text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        fontweight="bold",
    )


def pretty_task(task: str) -> str:
    return {"push": "Push", "strike": "Strike", "drawer": "Drawer"}[task]


def save_figure(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)


def load_bootstrap() -> pd.DataFrame:
    return pd.read_csv(PROBE / "results" / "bootstrap_cis.csv")


def load_transfer() -> pd.DataFrame:
    return pd.read_csv(
        PROBE / "trajectory_analysis_B" / "results" / "stats" / "cross_task_transfer.csv"
    )


def mean_r2_by_layer(csv_path: Path, layer_col: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df.groupby(layer_col, as_index=False)["r2"].mean()


def make_fig1_f5_hierarchy(ci: pd.DataFrame) -> None:
    sub = ci[ci["claim"] == "F5_delta_r2"].copy()
    tasks = ["push", "strike"]
    fig, axes = plt.subplots(2, 1, figsize=(3.25, 4.55), sharex=True, constrained_layout=True)

    y_max = float(sub["ci_hi"].max()) + 0.03
    for i, task in enumerate(tasks):
        ax = axes[i]
        task_df = sub[sub["task"] == task]
        for target in TARGET_ORDER:
            g = task_df[task_df["target"] == target].sort_values("layer")
            ax.plot(g["layer"], g["mean"], color=TARGET_COLORS[target], label=TARGET_LABELS[target])
            ax.fill_between(
                g["layer"],
                g["ci_lo"],
                g["ci_hi"],
                color=TARGET_COLORS[target],
                alpha=0.20,
                linewidth=0,
            )
        ax.axhline(0.0, color="0.35", linestyle="--", linewidth=0.8)
        ax.set_ylim(-0.02, y_max)
        ax.set_xlim(0, 23)
        ax.set_ylabel(r"$\Delta R^2$ (clean $-$ shuffled)")
        ax.set_xticks([0, 5, 10, 15, 20, 23])
        ax.grid(axis="y", color="0.9", linewidth=0.6)
        panel_label(ax, f"({chr(97+i)}) {pretty_task(task)}")
    axes[-1].set_xlabel("Layer")

    handles = [Line2D([0], [0], color=TARGET_COLORS[t], lw=1.4) for t in TARGET_ORDER]
    labels = [TARGET_LABELS[t] for t in TARGET_ORDER]
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.54, 1.01), ncol=2)
    save_figure(fig, "fig1_f5_hierarchy.pdf")


def make_fig2_f4a_modulation(ci: pd.DataFrame) -> None:
    sub = ci[ci["claim"] == "F4A_high_minus_low"].copy()
    configs = [
        ("strike", "object_0_static_friction", "(a) Strike static friction"),
        ("push", "object_0_mass", "(b) Push object mass"),
    ]
    targets = ["contact_flag", "contact_force_log1p_mag"]
    fig, axes = plt.subplots(2, 1, figsize=(3.25, 4.2), sharex=True, constrained_layout=True)

    for ax, (task, param, label) in zip(axes, configs):
        g_task = sub[(sub["task"] == task) & (sub["param"] == param)]
        for target in targets:
            g = g_task[g_task["target"] == target].sort_values("layer")
            ax.plot(g["layer"], g["mean"], color=TARGET_COLORS[target], label=TARGET_LABELS[target])
            ax.fill_between(
                g["layer"],
                g["ci_lo"],
                g["ci_hi"],
                color=TARGET_COLORS[target],
                alpha=0.20,
                linewidth=0,
            )
        ax.axhline(0.0, color="0.35", linestyle="--", linewidth=0.8)
        ax.set_xlim(0, 23)
        ax.set_xticks([0, 5, 10, 15, 20, 23])
        ax.set_ylabel(r"$\Delta R^2$ (High $-$ Low)")
        ax.grid(axis="y", color="0.9", linewidth=0.6)
        panel_label(ax, label)
    axes[-1].set_xlabel("Layer")
    handles = [Line2D([0], [0], color=TARGET_COLORS[t], lw=1.4) for t in targets]
    labels = [TARGET_LABELS[t] for t in targets]
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.52, 1.01), ncol=2)
    save_figure(fig, "fig2_f4a_modulation.pdf")


def format_cell(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def annotate_heatmap(ax: plt.Axes, data: np.ndarray, *, text_override: np.ndarray | None = None) -> None:
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            raw = data[i, j]
            text = format_cell(raw) if text_override is None else str(text_override[i, j])
            ax.text(j, i, text, ha="center", va="center", fontsize=6.5, color="black")


def make_fig3_transfer(transfer: pd.DataFrame) -> None:
    tasks = ["push", "strike", "drawer"]
    pair_order = [
        ("push", "strike"),
        ("strike", "push"),
        ("push", "drawer"),
        ("drawer", "push"),
        ("strike", "drawer"),
        ("drawer", "strike"),
    ]
    within = (
        transfer.groupby(["tgt_task", "target", "layer"], as_index=False)["within_tgt_r2"]
        .mean()
        .groupby(["tgt_task", "target"], as_index=False)["within_tgt_r2"]
        .max()
    )
    gap = (
        transfer.groupby(["src_task", "tgt_task", "target"], as_index=False)["gap"]
        .min()
    )

    mat_within = np.zeros((len(TARGET_ORDER), len(tasks)))
    for i, target in enumerate(TARGET_ORDER):
        for j, task in enumerate(tasks):
            val = within[(within["tgt_task"] == task) & (within["target"] == target)]["within_tgt_r2"]
            mat_within[i, j] = float(val.iloc[0]) if not val.empty else np.nan

    mat_gap = np.zeros((len(TARGET_ORDER), len(pair_order)))
    for i, target in enumerate(TARGET_ORDER):
        for j, (src, tgt) in enumerate(pair_order):
            val = gap[
                (gap["src_task"] == src) & (gap["tgt_task"] == tgt) & (gap["target"] == target)
            ]["gap"]
            mat_gap[i, j] = float(val.iloc[0]) if not val.empty else np.nan

    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.55), constrained_layout=True)

    im0 = axes[0].imshow(mat_within, aspect="auto", cmap="Blues", vmin=0.0, vmax=1.0)
    axes[0].set_xticks(range(len(tasks)))
    axes[0].set_xticklabels([pretty_task(t) for t in tasks])
    axes[0].set_yticks(range(len(TARGET_ORDER)))
    axes[0].set_yticklabels([TARGET_SHORT[t] for t in TARGET_ORDER])
    panel_label(axes[0], "(a) Best within-task $R^2$")
    annotate_heatmap(axes[0], mat_within)
    cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.03)
    cbar0.ax.tick_params(labelsize=6.5)

    clipped = np.clip(mat_gap, -0.2, 5.0)
    norm = mcolors.TwoSlopeNorm(vmin=-0.2, vcenter=0.5, vmax=5.0)
    im1 = axes[1].imshow(clipped, aspect="auto", cmap="viridis", norm=norm)
    axes[1].set_xticks(range(len(pair_order)))
    axes[1].set_xticklabels([f"{s[:1].upper()}→{t[:1].upper()}" for s, t in pair_order])
    axes[1].set_yticks(range(len(TARGET_ORDER)))
    axes[1].set_yticklabels([TARGET_SHORT[t] for t in TARGET_ORDER])
    panel_label(axes[1], "(b) Min transfer gap over layers")
    annotate_heatmap(axes[1], mat_gap)
    cbar1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.03)
    cbar1.set_label("Gap (clipped at 5)", fontsize=7)
    cbar1.ax.tick_params(labelsize=6.5)

    save_figure(fig, "fig3_transfer.pdf")


def load_vjepa_best_curve(task: str) -> tuple[np.ndarray, np.ndarray]:
    best_by_layer = None
    for target in ["ee_acceleration", "contact_flag", "contact_force_log1p_mag"]:
        df = mean_r2_by_layer(PROBE / "results" / task / "variant_A" / f"{target}.csv", "layer")
        arr = df.sort_values("layer")["r2"].to_numpy()
        best_by_layer = arr if best_by_layer is None else np.maximum(best_by_layer, arr)
    x = np.arange(len(best_by_layer)) / (len(best_by_layer) - 1)
    return x, best_by_layer


def load_r3m_best_curve(task: str) -> tuple[np.ndarray, np.ndarray]:
    summary = pd.read_csv(PROBE / "results" / task / "r3m" / "_summary.csv")
    stages = [f"stage_{i}" for i in range(5)]
    best = []
    for stage in stages:
        stage_rows = summary[summary["stage"] == stage]
        best.append(float(stage_rows["r2_mean"].max()))
    x = np.arange(len(best)) / (len(best) - 1)
    return x, np.asarray(best)


def make_fig4_r3m_vs_vjepa() -> None:
    fig, ax = plt.subplots(figsize=(3.25, 2.55), constrained_layout=True)

    for task in ["push", "strike", "drawer"]:
        color = TASK_COLORS[task]
        xv, yv = load_vjepa_best_curve(task)
        xr, yr = load_r3m_best_curve(task)
        ax.plot(xv, yv, color=color, marker="o", markersize=2.8, linestyle="-")
        ax.plot(xr, yr, color=color, marker="s", markersize=2.8, linestyle="--")

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 0.95)
    ax.set_xlabel("Normalized depth")
    ax.set_ylabel("Best target $R^2$")
    ax.grid(axis="y", color="0.9", linewidth=0.6)
    panel_label(ax, "(a) Best-of-target depth profile")

    task_handles = [
        Line2D([0], [0], color=TASK_COLORS[t], marker="o", linestyle="-", label=pretty_task(t))
        for t in ["push", "strike", "drawer"]
    ]
    model_handles = [
        Line2D([0], [0], color="black", marker="o", linestyle="-", label="V-JEPA A"),
        Line2D([0], [0], color="black", marker="s", linestyle="--", label="R3M"),
    ]
    leg1 = ax.legend(handles=task_handles, loc="upper left", ncol=1)
    ax.add_artist(leg1)
    ax.legend(handles=model_handles, loc="lower right", ncol=1)

    save_figure(fig, "fig4_r3m_vs_vjepa.pdf")


def main() -> None:
    configure_matplotlib()
    ci = load_bootstrap()
    transfer = load_transfer()
    make_fig1_f5_hierarchy(ci)
    make_fig2_f4a_modulation(ci)
    make_fig3_transfer(transfer)
    make_fig4_r3m_vs_vjepa()
    for name in [
        "fig1_f5_hierarchy.pdf",
        "fig2_f4a_modulation.pdf",
        "fig3_transfer.pdf",
        "fig4_r3m_vs_vjepa.pdf",
    ]:
        print(OUT / name)


if __name__ == "__main__":
    main()
