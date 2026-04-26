"""Per-window target builder. One target per window's last frame t.

Per spec § 4 + user instruction: finite-diff acceleration is the unified path
across all 6 tasks (avoids native-vs-derived distribution shift). We still
validate finite-diff vs stored GT acceleration where available (push, strike
for object; all six tasks for ee) before main sweep — gating threshold is
mean abs err < 5% of target std (config: thresholds.finite_diff_anchor_mae_frac).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .dataset import (
    parquet_for_episode,
    parquet_episode_ids,
    task_dt,
    task_object_keys,
    windows_for_T,
)
from .io import load_common, load_tasks


DIRECTION_MASK_TOL = 1e-4


def _stack(df: pd.DataFrame, col: str) -> np.ndarray:
    """Convert a column-of-arrays parquet field to [T, D] float32."""
    return np.stack(df[col].to_list()).astype(np.float32)


def _finite_diff(arr: np.ndarray, dt: float) -> np.ndarray:
    """Central diff with forward/backward at edges. Returns same shape as arr."""
    out = np.empty_like(arr)
    if arr.shape[0] >= 3:
        out[1:-1] = (arr[2:] - arr[:-2]) / (2.0 * dt)
        out[0] = (arr[1] - arr[0]) / dt
        out[-1] = (arr[-1] - arr[-2]) / dt
    elif arr.shape[0] == 2:
        out[0] = (arr[1] - arr[0]) / dt
        out[-1] = out[0]
    else:
        out[:] = 0.0
    return out


def _derive_kinematics(pos: np.ndarray, vel_native: np.ndarray, dt: float) -> dict[str, np.ndarray]:
    """Compute speed, direction, acceleration (finite-diff of velocity), accel_mag."""
    speed = np.linalg.norm(vel_native, axis=1, keepdims=False)
    dir_ = np.where(
        speed[:, None] > DIRECTION_MASK_TOL,
        vel_native / np.clip(speed[:, None], 1e-12, None),
        np.nan,
    )
    accel = _finite_diff(vel_native, dt).astype(np.float32)
    accel_mag = np.linalg.norm(accel, axis=1, keepdims=False)
    return {
        "position": pos.astype(np.float32),
        "velocity": vel_native.astype(np.float32),
        "speed": speed.astype(np.float32),
        "direction": dir_.astype(np.float32),
        "acceleration": accel.astype(np.float32),
        "accel_mag": accel_mag.astype(np.float32),
    }


def build_episode_targets(task: str, episode_id: int) -> dict[str, np.ndarray]:
    """Load an episode parquet and produce per-window targets at t_last for both EE
    and (if applicable) object. Returns dict with t_last, ee_*, obj_*, plus
    per-episode validation stats under `_val_*` (computed on the *full* episode
    trajectory before windowing — these are the correct anchors)."""
    df = pd.read_parquet(parquet_for_episode(task, episode_id))
    T = len(df)
    dt = task_dt(task)

    ee_pos = _stack(df, "physics_gt.ee_position")
    ee_vel = _stack(df, "physics_gt.ee_velocity")
    ee_kin = _derive_kinematics(ee_pos, ee_vel, dt)

    # Per-episode validation anchors (computed on full trajectory before windowing).
    val: dict[str, float] = {}
    v_fd = _finite_diff(ee_pos, dt)
    val["ee_vel_mae"] = float(np.mean(np.abs(v_fd - ee_vel)))
    val["ee_vel_std"] = float(np.std(ee_vel))
    if "physics_gt.ee_acceleration" in df.columns:
        ee_acc_native = _stack(df, "physics_gt.ee_acceleration")
        val["ee_acc_diff_mae"] = float(np.mean(np.abs(ee_kin["acceleration"] - ee_acc_native)))
        val["ee_acc_native_std"] = float(np.std(ee_acc_native))

    out: dict[str, np.ndarray] = {}
    out["t_last"] = windows_for_T(T)
    if out["t_last"].size == 0:
        return {"t_last": out["t_last"]}

    sel = out["t_last"]
    for k, v in ee_kin.items():
        out[f"ee_{k}"] = v[sel]

    obj_keys = task_object_keys(task)
    if obj_keys is not None:
        pos_c, vel_c, acc_c = obj_keys
        obj_pos = _stack(df, pos_c)
        obj_vel = _stack(df, vel_c)
        obj_kin = _derive_kinematics(obj_pos, obj_vel, dt)
        for k, v in obj_kin.items():
            out[f"obj_{k}"] = v[sel]
        if acc_c is not None and acc_c in df.columns:
            obj_acc_native = _stack(df, acc_c)
            val["obj_acc_diff_mae"] = float(np.mean(np.abs(obj_kin["acceleration"] - obj_acc_native)))
            val["obj_acc_native_std"] = float(np.std(obj_acc_native))

    out["episode_id"] = np.full((sel.size,), episode_id, dtype=np.int32)
    out["_val"] = np.array([val], dtype=object)  # carries dict through aggregation
    return out


def aggregate_task(
    task: str, episode_ids: list[int] | None = None
) -> tuple[dict[str, np.ndarray], list[dict[str, float]]]:
    """Concatenate per-episode targets into per-task arrays plus per-episode val stats.

    Returns:
      (windowed: {ee_*, obj_*, t_last, episode_id},
       per_ep_val: list of dicts with ee_vel_mae, ee_acc_diff_mae, obj_acc_diff_mae, ...)
    """
    eps = episode_ids if episode_ids is not None else parquet_episode_ids(task)
    parts: list[dict[str, np.ndarray]] = []
    val_list: list[dict[str, float]] = []
    for ep in eps:
        d = build_episode_targets(task, ep)
        if d.get("t_last", np.empty(0)).size == 0:
            continue
        if "_val" in d:
            val_list.append(d.pop("_val")[0])
        parts.append(d)
    keys = parts[0].keys()
    win = {k: np.concatenate([p[k] for p in parts if k in p], axis=0) for k in keys}
    return win, val_list


def validate_finite_diff(task: str, val_list: list[dict[str, float]]) -> dict[str, float]:
    """Aggregate per-episode validation stats into a task summary.

    Per spec note (revised 2026-04-25): native `physics_gt.<entity>_acceleration`
    is the Isaac-Lab body acceleration (physics engine) — NOT the time
    derivative of stored velocity. Velocity IS consistent with finite_diff(pos)
    at ~2% on push/strike. We therefore gate on *velocity consistency*; the
    native-vs-finite-diff acceleration disagreement is logged but informational.
    User directive: "use finite-diff uniformly to avoid distribution shift
    between native vs derived".
    """
    common = load_common()
    threshold = float(common["thresholds"]["finite_diff_anchor_mae_frac"])
    out: dict[str, float] = {"threshold": threshold}

    # Pooled fraction-of-std style aggregation: sum(mae) / sum(std) across episodes.
    s_vel_mae = sum(v["ee_vel_mae"] for v in val_list)
    s_vel_std = sum(v["ee_vel_std"] for v in val_list) + 1e-12
    out["ee_vel_consistency_mae_frac"] = s_vel_mae / s_vel_std

    if any("ee_acc_diff_mae" in v for v in val_list):
        s_acc_mae = sum(v.get("ee_acc_diff_mae", 0.0) for v in val_list)
        s_acc_std = sum(v.get("ee_acc_native_std", 0.0) for v in val_list) + 1e-12
        out["ee_acc_native_vs_finitediff_mae_frac"] = s_acc_mae / s_acc_std

    if any("obj_acc_diff_mae" in v for v in val_list):
        s_oacc_mae = sum(v.get("obj_acc_diff_mae", 0.0) for v in val_list)
        s_oacc_std = sum(v.get("obj_acc_native_std", 0.0) for v in val_list) + 1e-12
        out["obj_acc_native_vs_finitediff_mae_frac"] = s_oacc_mae / s_oacc_std

    out["pass"] = float(out["ee_vel_consistency_mae_frac"] < threshold)
    return out


def task_target_keys(task: str) -> list[str]:
    """Per-task target list per spec § 4."""
    keys = ["ee_position", "ee_velocity", "ee_speed", "ee_direction", "ee_acceleration", "ee_accel_mag"]
    if load_tasks()[task]["has_object"]:
        keys += ["obj_position", "obj_velocity", "obj_speed", "obj_direction", "obj_acceleration", "obj_accel_mag"]
    return keys
