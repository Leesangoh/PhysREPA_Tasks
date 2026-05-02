#!/usr/bin/env python3
"""Bootstrap CI runner for headline claims.

This script centralizes confidence-interval confirmation for a small set of
paper-facing claims using the statistical helpers in ``probe/utils/stats.py``.

Implemented claims:
1. F5 frame-shuffle delta R²:
   unshuffled - shuffled, paired over CV folds per (task, target, layer)
2. F4-A physics-condition bin gap:
   high - low, paired over CV folds IF a fold-level sidecar exists
3. F2 pre/during/post differences:
   during - pre and during - post, paired over CV folds IF a fold-level sidecar exists
4. Cross-task transfer gap:
   bootstrap over fold-level gap IF a fold-level sidecar exists

If a required input file is missing, or only aggregate means exist without
fold-level rows, the claim is skipped and logged. This preserves rigor: we do
not fabricate bootstrap samples from summary statistics.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROBE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROBE_ROOT))

from utils.stats import bootstrap_diff_ci


RESULTS_ROOT = PROBE_ROOT / "results"
TRAJ_B_ROOT = PROBE_ROOT / "trajectory_analysis_B" / "results" / "stats"
OUT_CSV = RESULTS_ROOT / "bootstrap_cis.csv"
LOG_PATH = RESULTS_ROOT / "bootstrap_cis.log"

F5_TASKS = ["push", "strike"]
F5_TARGETS = ["ee_velocity", "ee_acceleration", "contact_flag", "contact_force_log1p_mag"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap CIs for headline claims.")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Compute only a single push/layer row for F5 and print.")
    return parser.parse_args()


def log(msg: str) -> None:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(msg.rstrip() + "\n")


def flush_rows(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df[
            [
                "claim",
                "task",
                "target",
                "layer",
                "src_task",
                "tgt_task",
                "param",
                "n_episodes_or_folds",
                "mean",
                "ci_lo",
                "ci_hi",
                "p_value",
                "method",
            ]
        ]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)


def safe_str(x: str | None) -> str:
    return "" if x is None else str(x)


def make_row(
    *,
    claim: str,
    mean: float,
    ci_lo: float,
    ci_hi: float,
    p_value: float,
    method: str,
    n_episodes_or_folds: int,
    task: str | None = None,
    target: str | None = None,
    layer: int | None = None,
    src_task: str | None = None,
    tgt_task: str | None = None,
    param: str | None = None,
) -> dict:
    return {
        "claim": claim,
        "task": safe_str(task),
        "target": safe_str(target),
        "layer": "" if layer is None else int(layer),
        "src_task": safe_str(src_task),
        "tgt_task": safe_str(tgt_task),
        "param": safe_str(param),
        "n_episodes_or_folds": int(n_episodes_or_folds),
        "mean": float(mean),
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "p_value": float(p_value),
        "method": method,
    }


def load_fold_r2_csv(path: Path, required_cols: set[str] | None = None) -> pd.DataFrame | None:
    if not path.exists():
        log(f"[10] missing file: {path}")
        return None
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        log(f"[10] failed reading {path}: {exc}")
        return None
    needed = required_cols if required_cols is not None else {"fold", "r2"}
    if not needed.issubset(set(df.columns)):
        log(f"[10] missing required fold-level columns in {path}: need {sorted(needed)}")
        return None
    return df


def align_paired_fold_arrays(
    a_df: pd.DataFrame,
    b_df: pd.DataFrame,
    *,
    layer: int,
    extra_filters_a: dict | None = None,
    extra_filters_b: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    extra_filters_a = extra_filters_a or {}
    extra_filters_b = extra_filters_b or {}
    a_sub = a_df[a_df["layer"] == layer].copy()
    b_sub = b_df[b_df["layer"] == layer].copy()
    for k, v in extra_filters_a.items():
        a_sub = a_sub[a_sub[k] == v]
    for k, v in extra_filters_b.items():
        b_sub = b_sub[b_sub[k] == v]
    if a_sub.empty or b_sub.empty:
        return np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.float64)
    merged = a_sub[["fold", "r2"]].rename(columns={"r2": "r2_a"}).merge(
        b_sub[["fold", "r2"]].rename(columns={"r2": "r2_b"}),
        on="fold",
        how="inner",
    ).sort_values("fold")
    if merged.empty:
        return np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.float64)
    return merged["r2_a"].to_numpy(dtype=np.float64), merged["r2_b"].to_numpy(dtype=np.float64)


def candidate_sidecars(base_path: Path) -> list[Path]:
    stem = base_path.stem
    return [
        base_path,
        base_path.with_name(f"{stem}__per_fold.csv"),
        base_path.with_name(f"{stem}_per_fold.csv"),
    ]


def maybe_load_first_foldlevel(paths: list[Path]) -> pd.DataFrame | None:
    for path in paths:
        if path.exists():
            df = load_fold_r2_csv(path)
            if df is not None:
                return df
    if paths:
        log(f"[10] no fold-level sidecar found among: {', '.join(str(p) for p in paths)}")
    return None


def run_f5(rows: list[dict], args: argparse.Namespace) -> None:
    tasks = ["push"] if args.dry_run else F5_TASKS
    target_map = {task: (["ee_velocity"] if args.dry_run else F5_TARGETS) for task in tasks}
    layer_override = 11 if args.dry_run else None

    for task in tasks:
        for target in target_map[task]:
            un_path = RESULTS_ROOT / task / "variant_A" / f"{target}.csv"
            sh_path = RESULTS_ROOT / task / "variant_A_shuffled" / f"{target}.csv"
            un_df = load_fold_r2_csv(un_path)
            sh_df = load_fold_r2_csv(sh_path)
            if un_df is None or sh_df is None:
                log(f"[10:F5] skipping {task}/{target} due to missing fold-level un/shuffled CSV")
                continue
            layers = [layer_override] if layer_override is not None else sorted(set(un_df["layer"].tolist()) & set(sh_df["layer"].tolist()))
            for layer in layers:
                a, b = align_paired_fold_arrays(un_df, sh_df, layer=layer)
                if a.size == 0 or b.size == 0:
                    log(f"[10:F5] no aligned folds for {task}/{target}/L{layer}")
                    continue
                try:
                    out = bootstrap_diff_ci(a, b, paired=True, n_boot=args.n_boot, seed=args.seed)
                except Exception as exc:
                    log(f"[10:F5] bootstrap failed for {task}/{target}/L{layer}: {exc}")
                    continue
                rows.append(
                    make_row(
                        claim="F5_delta_r2",
                        task=task,
                        target=target,
                        layer=layer,
                        n_episodes_or_folds=a.size,
                        mean=out["diff_mean"],
                        ci_lo=out["ci_lo"],
                        ci_hi=out["ci_hi"],
                        p_value=out["p_value_one_tailed"],
                        method="bootstrap_diff_paired_over_folds",
                    )
                )
                flush_rows(rows)


def run_f4a(rows: list[dict], args: argparse.Namespace) -> None:
    split_dir = RESULTS_ROOT / "physics_condition_split"
    for base_path in sorted(split_dir.glob("*.csv")):
        if base_path.name.startswith("_"):
            continue
        if base_path.stem.endswith("_per_fold") or base_path.stem.endswith("__per_fold"):
            continue
        task_param = base_path.stem
        if "__" not in task_param:
            continue
        task, param = task_param.split("__", 1)
        fold_df = maybe_load_first_foldlevel(candidate_sidecars(base_path))
        if fold_df is None:
            log(f"[10:F4A] {task}/{param}: aggregate-only CSV present, no fold-level sidecar; skipped")
            continue
        needed = {"bin", "layer", "target", "fold", "r2"}
        if not needed.issubset(set(fold_df.columns)):
            log(f"[10:F4A] {task}/{param}: missing required cols {sorted(needed)} in fold-level CSV")
            continue
        for (target, layer), _ in fold_df.groupby(["target", "layer"]):
            a, b = align_paired_fold_arrays(
                fold_df,
                fold_df,
                layer=int(layer),
                extra_filters_a={"bin": "high", "target": target},
                extra_filters_b={"bin": "low", "target": target},
            )
            if a.size == 0 or b.size == 0:
                continue
            try:
                out = bootstrap_diff_ci(a, b, paired=True, n_boot=args.n_boot, seed=args.seed)
            except Exception as exc:
                log(f"[10:F4A] bootstrap failed for {task}/{param}/{target}/L{layer}: {exc}")
                continue
            rows.append(
                make_row(
                    claim="F4A_high_minus_low",
                    task=task,
                    target=str(target),
                    layer=int(layer),
                    param=param,
                    n_episodes_or_folds=a.size,
                    mean=out["diff_mean"],
                    ci_lo=out["ci_lo"],
                    ci_hi=out["ci_hi"],
                    p_value=out["p_value_one_tailed"],
                    method="bootstrap_diff_paired_over_folds",
                )
            )
            flush_rows(rows)


def run_f2(rows: list[dict], args: argparse.Namespace) -> None:
    base_path = TRAJ_B_ROOT / "phase_conditional.csv"
    fold_df = maybe_load_first_foldlevel(candidate_sidecars(base_path))
    if fold_df is None:
        log("[10:F2] phase_conditional has no fold-level sidecar; skipped")
        return
    needed = {"task", "layer", "target", "condition_type", "condition_value", "fold", "r2"}
    if not needed.issubset(set(fold_df.columns)):
        log(f"[10:F2] missing required cols {sorted(needed)} in fold-level CSV")
        return
    cond_df = fold_df[fold_df["condition_type"] == "contact_phase3"].copy()
    if cond_df.empty:
        log("[10:F2] no contact_phase3 rows in fold-level CSV")
        return
    pairs = [("during", "pre", "F2_during_minus_pre"), ("during", "post", "F2_during_minus_post")]
    for (task, target, layer), _ in cond_df.groupby(["task", "target", "layer"]):
        for left, right, claim_name in pairs:
            a, b = align_paired_fold_arrays(
                cond_df,
                cond_df,
                layer=int(layer),
                extra_filters_a={"task": task, "target": target, "condition_value": left},
                extra_filters_b={"task": task, "target": target, "condition_value": right},
            )
            if a.size == 0 or b.size == 0:
                continue
            try:
                out = bootstrap_diff_ci(a, b, paired=True, n_boot=args.n_boot, seed=args.seed)
            except Exception as exc:
                log(f"[10:F2] bootstrap failed for {task}/{target}/L{layer}/{left}-{right}: {exc}")
                continue
            rows.append(
                make_row(
                    claim=claim_name,
                    task=str(task),
                    target=str(target),
                    layer=int(layer),
                    n_episodes_or_folds=a.size,
                    mean=out["diff_mean"],
                    ci_lo=out["ci_lo"],
                    ci_hi=out["ci_hi"],
                    p_value=out["p_value_one_tailed"],
                    method="bootstrap_diff_paired_over_folds",
                )
            )
            flush_rows(rows)


def run_transfer(rows: list[dict], args: argparse.Namespace) -> None:
    sidecar_path = TRAJ_B_ROOT / "cross_task_transfer_per_fold.csv"
    fold_df = load_fold_r2_csv(
        sidecar_path,
        required_cols={"src_task", "tgt_task", "layer", "target", "fold", "transfer_r2", "within_tgt_r2", "gap"},
    )
    if fold_df is None:
        log("[10:transfer] cross_task_transfer_per_fold sidecar missing or invalid; skipped")
        return
    for (src_task, tgt_task, target, layer), sub in fold_df.groupby(["src_task", "tgt_task", "target", "layer"]):
        gap = sub.sort_values("fold")["gap"].to_numpy(dtype=np.float64)
        gap = gap[np.isfinite(gap)]
        if gap.size == 0:
            continue
        try:
            out = bootstrap_diff_ci(gap, np.zeros_like(gap), paired=True, n_boot=args.n_boot, seed=args.seed)
        except Exception as exc:
            log(f"[10:transfer] bootstrap failed for {src_task}->{tgt_task}/{target}/L{layer}: {exc}")
            continue
        rows.append(
            make_row(
                claim="transfer_gap",
                src_task=str(src_task),
                tgt_task=str(tgt_task),
                target=str(target),
                layer=int(layer),
                n_episodes_or_folds=gap.size,
                mean=out["diff_mean"],
                ci_lo=out["ci_lo"],
                ci_hi=out["ci_hi"],
                p_value=out["p_value_one_tailed"],
                method="bootstrap_gap_vs_zero_over_folds",
            )
        )
        flush_rows(rows)


def main() -> None:
    args = parse_args()
    rows: list[dict] = []
    flush_rows(rows)
    run_f5(rows, args)
    if not args.dry_run:
        run_f4a(rows, args)
        run_f2(rows, args)
        run_transfer(rows, args)
    if args.dry_run:
        df = pd.DataFrame(rows)
        print(df.to_string(index=False) if not df.empty else "[10] dry-run produced 0 rows", flush=True)


if __name__ == "__main__":
    main()
