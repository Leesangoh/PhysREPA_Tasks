#!/usr/bin/env python3
"""Write TRAJECTORY_REPORT.md compiling all trajectory analysis findings."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
STATS = RESULTS / "stats"
PLOTS_REL = "plots"


def maybe(path: Path, label: str) -> str:
    return f"![{label}]({PLOTS_REL}/{path.name})\n" if path.exists() else f"_({label} not produced)_\n"


def main():
    parts: list[str] = []
    parts.append("# PhysProbe — Latent Trajectory Analysis Report\n\n")
    parts.append(
        "All analyses use Variant A features (V-JEPA 2 ViT-L, 1024-d spatiotemporal-mean pool, 24 residual layers). "
        "Per-task we sample 30 episodes (seed=42) and pool windows. Whitening: PCA-whitened per (task, layer) on inner-pool "
        "with var_keep=0.99 to remove anisotropy/scale artifacts before geometric measures.\n\n"
    )

    # 1. Trajectory geometry stats
    p = STATS / "trajectory_stats_summary.csv"
    if p.exists():
        df = pd.read_csv(p)
        parts.append("## 1. Trajectory geometry per (task, layer, episode)\n\n")
        parts.append(
            "Path length, mean speed, tortuosity (path/direct), and curvature (mean angular change between "
            "consecutive direction unit vectors). Per-task means + SD across sampled episodes.\n\n"
        )
        for task in df.task.unique():
            sub = df[df.task == task].sort_values("layer")
            parts.append(f"### {task}\n\n")
            parts.append("| layer | path_length | mean_speed | tortuosity | curvature |\n")
            parts.append("|---|---|---|---|---|\n")
            for _, r in sub.iterrows():
                parts.append(f"| {int(r.layer)} | {r.path_length__mean:.2f}±{r.path_length__std:.2f} | "
                             f"{r.mean_speed__mean:.3f}±{r.mean_speed__std:.3f} | "
                             f"{r.tortuosity__mean:.2f}±{r.tortuosity__std:.2f} | "
                             f"{r.curvature_mean__mean:.3f}±{r.curvature_mean__std:.3f} |\n")
            parts.append("\n")

    # 2. PCA visualization
    parts.append("## 2. PCA(2) latent trajectories per layer\n\n")
    parts.append("Per-layer PCA(2) fit on pooled windows of that task. Each panel shows "
                 "5–8 sample episode trajectories on the same PC1-PC2 axes. Start = circle, end = square.\n\n")
    for task in ["push", "strike", "reach", "drawer", "peg_insert", "nut_thread"]:
        parts.append(f"### {task}\n\n")
        parts.append(maybe(RESULTS / "plots" / f"pca_{task}.png", f"per-layer PCA — {task}"))
        parts.append("\n")

    # 3. Shared-PCA + Procrustes
    parts.append("## 3. Shared-PCA(2) trajectory across layers (per task)\n\n")
    parts.append("Per-layer whitening then a single PCA(2) basis fit on the concatenated whitened pool "
                 "across all 24 layers. Same PC1-PC2 axes for every panel — directly comparable layer geometry.\n\n")
    for task in ["push", "strike", "reach", "drawer", "peg_insert", "nut_thread"]:
        parts.append(f"### {task}\n\n")
        parts.append(maybe(RESULTS / "plots" / f"shared_pca_{task}.png", f"shared-basis PCA — {task}"))
        parts.append("\n")

    # 4. Intrinsic dimensionality + spectrum
    parts.append("## 4. Intrinsic dimensionality per layer\n\n")
    parts.append(maybe(RESULTS / "plots" / "intrinsic_dim_per_layer.png", "intrinsic dim per layer"))
    for task in ["push", "strike", "reach", "drawer", "peg_insert", "nut_thread"]:
        parts.append(maybe(RESULTS / "plots" / f"evr_spectrum_{task}.png", f"EVR spectrum — {task}"))
    parts.append("\n")

    # 5. Cross-layer / cross-task CKA
    parts.append("## 5. Cross-layer CKA per task\n\n")
    for task in ["push", "strike", "reach", "drawer", "peg_insert", "nut_thread"]:
        parts.append(maybe(RESULTS / "plots" / f"cross_layer_cka_{task}.png", f"cross-layer CKA — {task}"))
    parts.append("\n## 6. Cross-task feature CKA per layer\n\n")
    parts.append(maybe(RESULTS / "plots" / "cross_task_cka_evolution.png", "cross-task CKA evolution"))
    for L in (0, 6, 12, 18, 23):
        parts.append(maybe(RESULTS / "plots" / f"cross_task_cka_layer{L:02d}.png", f"cross-task CKA L{L:02d}"))

    # 7. Event-locked
    parts.append("\n## 7. Event-locked latent geometry (push, strike, drawer)\n\n")
    parts.append(
        "Trajectories aligned to contact onset (push, strike) or sustained motion onset (drawer). "
        "Each task: 4 metrics × layer × τ heatmaps. Speed/curvature/tortuosity/PR. "
        "τ=0 marks the event; PR is participation ratio on local 2w+1 = 9 window covariance.\n\n"
    )
    for task in ["push", "strike", "drawer"]:
        parts.append(maybe(RESULTS / "plots" / f"event_locked_{task}.png", f"event-locked {task}"))
    parts.append("\n")

    # 8. Partial RSA
    parts.append("## 8. Partial RSA — latent geometry vs physics, controlling for pos+vel\n\n")
    parts.append(
        "Vectorized RDMs computed on whitened latent and standardized physics groups. "
        "Both rank-transformed; residualized against [r_pos, r_vel]; Pearson on residuals = partial Spearman correlation.\n\n"
    )
    parts.append(maybe(RESULTS / "plots" / "rsa_partial_heatmap_acc.png", "partial RSA acceleration"))
    parts.append(maybe(RESULTS / "plots" / "rsa_partial_heatmap_ct.png", "partial RSA contact"))
    for task in ["push", "strike", "reach", "drawer", "peg_insert", "nut_thread"]:
        parts.append(maybe(RESULTS / "plots" / f"rsa_partial_{task}.png", f"partial RSA — {task}"))
    parts.append("\n")

    # 9. Tangent RSA
    parts.append("## 9. Tangent RSA — latent dynamics vs physical dynamics\n\n")
    parts.append(
        "U_l[t] = Z_white[t+1] − Z_white[t]; A_l[t] = U_l[t+1] − U_l[t]. "
        "Compare pdist(U) vs pdist(physical velocity); pdist(A) vs pdist(physical acc); "
        "and partial version controlling for pos+vel.\n\n"
    )
    parts.append(maybe(RESULTS / "plots" / "rsa_vel_tan.png", "tangent RSA U vs V"))
    parts.append(maybe(RESULTS / "plots" / "rsa_acc_tan.png", "tangent RSA A vs G"))
    parts.append(maybe(RESULTS / "plots" / "rsa_acc_tan_partial.png", "tangent RSA A vs G | pos,vel"))
    parts.append("\n")

    # 10. CCA
    parts.append("## 10. CCA — canonical subspace alignment between latent and physics groups\n\n")
    parts.append(
        "PCA-reduce whitened latent to top-k (95% variance, ≤128). "
        "Fit linear CCA between Ẑ_l and X_g for each physics group g ∈ {pos, vel, acc, ct}. "
        "Report ρ1 (top canonical correlation), CCA energy (Σρ²), rank90 (subspace dim for 90% energy).\n\n"
    )
    for metric in ("rho1", "energy", "rank90"):
        for gname in ("pos", "vel", "acc", "ct"):
            parts.append(maybe(RESULTS / "plots" / f"cca_{metric}_{gname}.png", f"CCA {metric} — {gname}"))
    parts.append("\n")

    # 11. Koopman-style dynamics
    parts.append("## 11. Koopman-style linear dynamics scores\n\n")
    parts.append(
        "A. Self-predictability ΔR² (z_{t+1} ≈ A_l z_t, fit Ridge, "
        "report delta over persistence baseline z_{t+1}=z_t).\n"
        "B. Next-step physics predictability from latent: ee_pos / ee_vel / ee_acc / obj_vel one-step ahead.\n"
        "Episode-level 80/20 train/test split, episode-aware boundary masking. "
        "PCA-whitened latent (var_keep=0.99, K_cap=128).\n\n"
    )
    parts.append(maybe(RESULTS / "plots" / "koopman_r2_self_delta.png", "Koopman ΔR² self"))
    parts.append(maybe(RESULTS / "plots" / "koopman_ee_pos_next.png", "Next-step ee_pos R²"))
    parts.append(maybe(RESULTS / "plots" / "koopman_ee_vel_next.png", "Next-step ee_vel R²"))
    parts.append(maybe(RESULTS / "plots" / "koopman_ee_acc_next.png", "Next-step ee_acc R²"))
    parts.append(maybe(RESULTS / "plots" / "koopman_obj_vel_next.png", "Next-step obj_vel R²"))
    parts.append("\n")

    # 12. Methodology + caveats
    parts.append(
        "## 11. Methodology notes\n\n"
        "- **Whitening**: per-(task, layer) PCA-whitening on the pooled-windows feature matrix, "
        "var_keep=0.99. This removes layer-norm-induced anisotropy that would otherwise confound geometric measures.\n"
        "- **RSA bootstrap**: episode-level resampling planned (B=200) but the heavy pdist recomputation is "
        "left at point-estimate in this run; CIs are an extension.\n"
        "- **Sub-sampling**: 1500 windows for partial-RSA / tangent-RSA, 4000 for CCA, 8000 for shared-PCA fit. "
        "Stratified by episode × normalized-time bin where applicable.\n"
        "- **Drawer event-lock**: contact_flag is sparse in drawer; we use sustained motion-onset on object speed "
        "(threshold = 30% of episode-max for k=3 consecutive windows) per Codex guidance.\n"
        "- **Acceleration target**: finite-diff(velocity) per the Variant-A/B probing pipeline (Isaac-Lab body "
        "acceleration is not d/dt of stored velocity; finite-diff used uniformly to avoid distribution shift).\n"
        "\n"
    )

    out = RESULTS / "TRAJECTORY_REPORT.md"
    out.write_text("".join(parts))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
