#!/usr/bin/env python3
"""Window-level probing split by randomized physics conditions.

For each (task, physics param), split cached episodes into low/med/high terciles
by the episode-level parameter value, then run Variant A layerwise Ridge probes
within each bin.

Outputs:
  probe/results/physics_condition_split/<task>__<param>.csv
with columns:
  bin, layer, target, r2_mean, r2_std, n_episodes_in_bin, n_windows_in_bin, q33, q67
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

PROBE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROBE_ROOT))

from utils.io import load_common, list_cached_episodes, load_episode_features, load_targets


OUT_DIR = PROBE_ROOT / "results" / "physics_condition_split"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUT_DIR / "_log.txt"

TASKS = ["push", "strike", "drawer", "peg_insert", "nut_thread"]
TARGETS = ["ee_velocity", "ee_acceleration", "contact_flag", "contact_force_log1p_mag"]
SEED = 42

PHYSICS_PARAMS = {
    "push": [
        "object_0_mass",
        "object_0_static_friction",
        "object_0_dynamic_friction",
        "surface_static_friction",
        "surface_dynamic_friction",
    ],
    "strike": [
        "object_0_mass",
        "object_0_static_friction",
        "object_0_dynamic_friction",
        "surface_static_friction",
        "surface_dynamic_friction",
        "object_0_restitution",
    ],
    "drawer": [
        "drawer_joint_damping",
    ],
    "peg_insert": [
        "peg_static_friction",
        "peg_dynamic_friction",
        "peg_mass",
        "hole_static_friction",
        "hole_dynamic_friction",
    ],
    "nut_thread": [
        "nut_static_friction",
        "nut_dynamic_friction",
        "nut_mass",
        "bolt_static_friction",
        "bolt_dynamic_friction",
    ],
    "reach": [],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe split by physics-condition terciles.")
    parser.add_argument("--task", default="all", help="all or comma-separated task list")
    parser.add_argument("--param", default="all", help="all or a single physics parameter name")
    parser.add_argument("--device", default="cpu", help="cpu or cuda:N for closed-form GPU Ridge")
    parser.add_argument("--dry-run", action="store_true", help="push/object_0_mass/30eps/layers 0,11,23/ee_velocity only")
    return parser.parse_args()


def log(msg: str) -> None:
    with open(LOG_PATH, "a") as f:
        f.write(msg.rstrip() + "\n")


def load_episode_param_map(task: str) -> dict[int, dict[str, float]]:
    """Read per-episode physics params from meta/episodes.jsonl."""
    params = PHYSICS_PARAMS.get(task, [])
    if not params:
        return {}
    common = load_common()
    path = Path(common["dataset_root"]) / task / "meta" / "episodes.jsonl"
    out: dict[int, dict[str, float]] = {}
    if not path.exists():
        log(f"[09] {task}: missing {path}")
        return out
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            ep = int(rec["episode_index"])
            out[ep] = {}
            for p in params:
                if p in rec:
                    try:
                        out[ep][p] = float(rec[p])
                    except Exception:
                        continue
    return out


def fit_group_ridge_with_folds(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    alpha: float = 1.0,
) -> tuple[float, float, list[float], list[dict[str, int | float]]]:
    """5-fold GroupKFold Ridge(alpha=1) with per-fold outputs (CPU sklearn path)."""
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.reshape(-1)
    if y.ndim == 1:
        ok = np.isfinite(y)
    else:
        ok = np.isfinite(y).all(axis=1)
    if ok.sum() < 100:
        return float("nan"), float("nan"), [], []
    X = X[ok]
    y = y[ok]
    groups = groups[ok]
    if np.unique(groups).size < 5:
        return float("nan"), float("nan"), [], []
    r2s: list[float] = []
    fold_meta: list[dict[str, int | float]] = []
    gkf = GroupKFold(n_splits=5)
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups=groups)):
        if tr.size < 50 or te.size < 20:
            continue
        Xtr = X[tr]
        Xte = X[te]
        ytr = y[tr]
        yte = y[te]
        mu_x = Xtr.mean(axis=0)
        sd_x = Xtr.std(axis=0) + 1e-9
        Xtr_n = (Xtr - mu_x) / sd_x
        Xte_n = (Xte - mu_x) / sd_x
        if ytr.ndim == 1:
            mu_y = ytr.mean()
            sd_y = ytr.std() + 1e-9
        else:
            mu_y = ytr.mean(axis=0)
            sd_y = ytr.std(axis=0) + 1e-9
        ytr_n = (ytr - mu_y) / sd_y
        model = Ridge(alpha=alpha)
        model.fit(Xtr_n, ytr_n)
        pred = model.predict(Xte_n) * sd_y + mu_y
        if yte.ndim == 1:
            r2 = float(r2_score(yte, pred))
        else:
            r2 = float(r2_score(yte, pred, multioutput="variance_weighted"))
        r2s.append(r2)
        fold_meta.append(
            {
                "fold": int(fold),
                "r2": r2,
                "n_train_episodes": int(np.unique(groups[tr]).size),
                "n_test_episodes": int(np.unique(groups[te]).size),
                "n_test_windows": int(te.size),
            }
        )
    if not r2s:
        return float("nan"), float("nan"), [], []
    return float(np.mean(r2s)), float(np.std(r2s)), r2s, fold_meta


def fit_group_ridge_with_folds_gpu(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    device: str,
    alpha: float = 1.0,
    n_splits: int = 5,
) -> tuple[float, float, list[float], list[dict[str, int | float]]]:
    """Closed-form Ridge on GPU via torch.linalg.solve, with CPU fallback on OOM.

    Matches the CPU path:
      - finite-row filtering
      - GroupKFold by episode
      - within-fold z-score
      - variance-weighted multioutput R²

    Note:
      We intentionally use float32 on GPU. On A6000-class GPUs, float64 solve
      is extremely slow and not justified here; empirical tolerance for this
      analysis is |ΔR²| ≲ 1e-3 relative to the CPU sklearn path.
    """
    try:
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.reshape(-1)
        if y.ndim == 1:
            ok = np.isfinite(y)
        else:
            ok = np.isfinite(y).all(axis=1)
        if ok.sum() < 100:
            return float("nan"), float("nan"), [], []
        X = X[ok]
        y = y[ok]
        groups = groups[ok]
        if np.unique(groups).size < n_splits:
            return float("nan"), float("nan"), [], []

        r2s: list[float] = []
        fold_meta: list[dict[str, int | float]] = []
        gkf = GroupKFold(n_splits=n_splits)
        torch_device = torch.device(device)
        eye_cache: dict[int, torch.Tensor] = {}

        for fold, (tr, te) in enumerate(gkf.split(X, y, groups=groups)):
            if tr.size < 50 or te.size < 20:
                continue
            Xtr = X[tr].astype(np.float32, copy=False)
            Xte = X[te].astype(np.float32, copy=False)
            ytr = y[tr].astype(np.float32, copy=False)
            yte = y[te].astype(np.float32, copy=False)

            mu_x = Xtr.mean(axis=0)
            sd_x = Xtr.std(axis=0) + 1e-9
            Xtr_n = (Xtr - mu_x) / sd_x
            Xte_n = (Xte - mu_x) / sd_x

            y_scalar = (ytr.ndim == 1)
            if y_scalar:
                mu_y = float(ytr.mean())
                sd_y = float(ytr.std() + 1e-9)
                ytr_n = ((ytr - mu_y) / sd_y)[:, None]
            else:
                mu_y = ytr.mean(axis=0)
                sd_y = ytr.std(axis=0) + 1e-9
                ytr_n = (ytr - mu_y) / sd_y

            Xtr_t = torch.as_tensor(Xtr_n, dtype=torch.float32, device=torch_device)
            Xte_t = torch.as_tensor(Xte_n, dtype=torch.float32, device=torch_device)
            ytr_t = torch.as_tensor(ytr_n, dtype=torch.float32, device=torch_device)

            F_dim = Xtr_t.shape[1]
            if F_dim not in eye_cache:
                eye_cache[F_dim] = torch.eye(F_dim, dtype=torch.float32, device=torch_device)
            eye = eye_cache[F_dim]
            xtx = Xtr_t.T @ Xtr_t
            xty = Xtr_t.T @ ytr_t
            w = torch.linalg.solve(xtx + alpha * eye, xty)
            pred_n = Xte_t @ w
            pred = pred_n.cpu().numpy()
            if y_scalar:
                pred = pred.reshape(-1) * sd_y + mu_y
                r2 = float(r2_score(yte, pred))
            else:
                pred = pred * sd_y + mu_y
                r2 = float(r2_score(yte, pred, multioutput="variance_weighted"))
            r2s.append(r2)
            fold_meta.append(
                {
                    "fold": int(fold),
                    "r2": r2,
                    "n_train_episodes": int(np.unique(groups[tr]).size),
                    "n_test_episodes": int(np.unique(groups[te]).size),
                    "n_test_windows": int(te.size),
                }
            )
            del Xtr_t, Xte_t, ytr_t, xtx, xty, w
            if torch_device.type == "cuda":
                torch.cuda.empty_cache()

        if not r2s:
            return float("nan"), float("nan"), [], []
        return float(np.mean(r2s)), float(np.std(r2s)), r2s, fold_meta
    except torch.cuda.OutOfMemoryError:
        log(f"[09] GPU OOM on device={device}; falling back to CPU Ridge")
        return fit_group_ridge_with_folds(X, y, groups, alpha=alpha)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            log(f"[09] GPU OOM runtime on device={device}; falling back to CPU Ridge")
            return fit_group_ridge_with_folds(X, y, groups, alpha=alpha)
        raise


def flush_csvs(
    agg_rows: list[dict],
    per_fold_rows: list[dict],
    agg_path: Path,
    per_fold_path: Path,
) -> None:
    pd.DataFrame(agg_rows).to_csv(agg_path, index=False)
    pd.DataFrame(per_fold_rows).to_csv(per_fold_path, index=False)


def build_target_lookup(tgt: dict[str, np.ndarray]) -> dict[tuple[int, int], int]:
    return {
        (int(ep), int(t_last)): i
        for i, (ep, t_last) in enumerate(zip(tgt["episode_id"].tolist(), tgt["t_last"].tolist()))
    }


def load_bin_payloads(
    task: str,
    episode_ids: list[int],
    tgt_lut: dict[tuple[int, int], int],
) -> list[dict]:
    """Load per-episode feature payloads once for a bin."""
    payloads: list[dict] = []
    for ep in episode_ids:
        try:
            d = load_episode_features(task, "A", ep)
            rows_idx = np.array([tgt_lut[(ep, int(t))] for t in d["t_last"]], dtype=np.int64)
            payloads.append(
                {
                    "episode_id": ep,
                    "feats": d["feats"],  # float16 on disk; keep as-is until layer slice
                    "rows_idx": rows_idx,
                }
            )
        except Exception as exc:
            log(f"[09] {task} ep={ep}: payload load skipped ({exc})")
            continue
    return payloads


def stack_layer_target(
    payloads: list[dict],
    tgt: dict[str, np.ndarray],
    layer: int,
    target: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    xs = []
    ys = []
    groups = []
    n_windows_total = 0
    for p in payloads:
        try:
            feats_l = p["feats"][:, layer, :].astype(np.float32, copy=False)
            y_ep = tgt[target][p["rows_idx"]]
            if y_ep.ndim == 2 and y_ep.shape[1] == 1:
                y_ep = y_ep.reshape(-1)
            xs.append(feats_l)
            ys.append(y_ep)
            groups.append(np.full(feats_l.shape[0], p["episode_id"], dtype=np.int64))
            n_windows_total += int(feats_l.shape[0])
        except Exception as exc:
            log(f"[09] stack failed target={target} layer={layer} ep={p['episode_id']}: {exc}")
            continue
    if not xs:
        return (
            np.empty((0, 1024), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            0,
        )
    X = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    g = np.concatenate(groups, axis=0)
    return X, y, g, n_windows_total


def task_list_from_args(args: argparse.Namespace) -> list[str]:
    if args.dry_run:
        return ["push"]
    if args.task == "all":
        return TASKS
    return [x.strip() for x in args.task.split(",") if x.strip()]


def param_list_for_task(task: str, args: argparse.Namespace) -> list[str]:
    if args.dry_run:
        return ["object_0_mass"] if task == "push" else []
    if args.param == "all":
        return PHYSICS_PARAMS.get(task, [])
    if args.param in PHYSICS_PARAMS.get(task, []):
        return [args.param]
    return []


def run_task_param(task: str, param: str, args: argparse.Namespace) -> None:
    tgt = load_targets(task)
    tgt_lut = build_target_lookup(tgt)
    ep_to_param = load_episode_param_map(task)
    eps = list_cached_episodes(task, "A")
    if args.dry_run:
        rng = np.random.default_rng(SEED)
        if len(eps) > 30:
            idx = rng.choice(len(eps), size=30, replace=False)
            eps = [eps[i] for i in sorted(idx.tolist())]

    vals = []
    eps_valid = []
    for ep in eps:
        if ep in ep_to_param and param in ep_to_param[ep] and np.isfinite(ep_to_param[ep][param]):
            vals.append(float(ep_to_param[ep][param]))
            eps_valid.append(ep)
    if len(eps_valid) < 15:
        log(f"[09] {task}/{param}: too few valid episodes ({len(eps_valid)})")
        return

    vals_arr = np.asarray(vals, dtype=np.float64)
    q33 = float(np.quantile(vals_arr, 0.33))
    q67 = float(np.quantile(vals_arr, 0.67))
    bins = {
        "low": [ep for ep in eps_valid if ep_to_param[ep][param] <= q33],
        "med": [ep for ep in eps_valid if q33 < ep_to_param[ep][param] <= q67],
        "high": [ep for ep in eps_valid if ep_to_param[ep][param] > q67],
    }

    layers = [0, 11, 23] if args.dry_run else list(range(24))
    targets = ["ee_velocity"] if args.dry_run else TARGETS
    rows: list[dict] = []
    per_fold_rows: list[dict] = []
    agg_path = OUT_DIR / f"{task}__{param}.csv"
    per_fold_path = OUT_DIR / f"{task}__{param}__per_fold.csv"

    for bin_name, ep_bin in bins.items():
        try:
            payloads = load_bin_payloads(task, ep_bin, tgt_lut)
        except Exception as exc:
            log(f"[09] {task}/{param}/{bin_name}: payload bin load failed ({exc})")
            continue

        for layer in layers:
            for target in targets:
                if target not in tgt:
                    continue
                try:
                    X, y, groups, n_windows = stack_layer_target(payloads, tgt, layer, target)
                    if X.shape[0] > 0:
                        if args.device.startswith("cuda"):
                            r2_mean, r2_std, _, fold_meta = fit_group_ridge_with_folds_gpu(
                                X, y, groups, device=args.device, alpha=1.0, n_splits=5
                            )
                            if args.dry_run:
                                cpu_mean, _, _, _ = fit_group_ridge_with_folds(X, y, groups, alpha=1.0)
                                if np.isfinite(r2_mean) and np.isfinite(cpu_mean):
                                    diff = abs(r2_mean - cpu_mean)
                                    log(
                                        f"[09 dry-run check] {task}/{param}/{bin_name}/L{layer}/{target}: "
                                        f"gpu={r2_mean:.6f} cpu={cpu_mean:.6f} |diff|={diff:.6f}"
                                    )
                        else:
                            r2_mean, r2_std, _, fold_meta = fit_group_ridge_with_folds(X, y, groups, alpha=1.0)
                    else:
                        r2_mean, r2_std, fold_meta = float("nan"), float("nan"), []
                    rows.append(
                        {
                            "bin": bin_name,
                            "layer": int(layer),
                            "target": target,
                            "r2_mean": r2_mean,
                            "r2_std": r2_std,
                            "n_episodes_in_bin": int(len(ep_bin)),
                            "n_windows_in_bin": int(n_windows),
                            "q33": q33,
                            "q67": q67,
                        }
                    )
                    for meta in fold_meta:
                        per_fold_rows.append(
                            {
                                "bin": bin_name,
                                "layer": int(layer),
                                "target": target,
                                "fold": int(meta["fold"]),
                                "r2": float(meta["r2"]),
                                "n_train_episodes": int(meta["n_train_episodes"]),
                                "n_test_episodes": int(meta["n_test_episodes"]),
                                "n_test_windows": int(meta["n_test_windows"]),
                            }
                        )
                except Exception as exc:
                    log(f"[09] {task}/{param}/{bin_name}/L{layer}/{target}: skipped ({exc})")
                    rows.append(
                        {
                            "bin": bin_name,
                            "layer": int(layer),
                            "target": target,
                            "r2_mean": float("nan"),
                            "r2_std": float("nan"),
                            "n_episodes_in_bin": int(len(ep_bin)),
                            "n_windows_in_bin": 0,
                            "q33": q33,
                            "q67": q67,
                        }
                    )
            if not args.dry_run:
                flush_csvs(rows, per_fold_rows, agg_path, per_fold_path)

    pd.DataFrame(rows).to_csv(agg_path, index=False)
    pd.DataFrame(per_fold_rows).to_csv(per_fold_path, index=False)
    print(f"[09_phys_split] wrote {agg_path} ({len(rows)} rows)", flush=True)


def main() -> None:
    args = parse_args()
    for task in task_list_from_args(args):
        for param in param_list_for_task(task, args):
            try:
                print(f"[09_phys_split] {task}/{param} starting", flush=True)
                run_task_param(task, param, args)
            except Exception as exc:
                log(f"[09] {task}/{param}: fatal-but-skipped ({exc})")
                continue


if __name__ == "__main__":
    main()
