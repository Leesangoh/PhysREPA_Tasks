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

    # =====================================================================
    # EXTENDED TARGETS — Tier-Contact, Tier-Rotational, Tier-Progress
    # Same probe methodology (window-aligned, per-frame t_last). Available
    # fields vary per task; we extract what's present.
    # =====================================================================

    # Tier-Rotational: ee_orientation, ee_angular_velocity (all 6 tasks)
    # IMPORTANT (per Codex): canonicalize quaternions — q and -q are the same
    # orientation, so we flip sign whenever q_w < 0 to put the rotation on a
    # single hemisphere. Without this the probe must learn an arbitrary
    # discontinuity and R² is artificially worse.
    def _canon_q(q):
        # q shape: [T, 4] with convention (w, x, y, z) per Isaac Lab
        flip = q[:, 0] < 0
        q[flip] = -q[flip]
        # Re-normalize defensively (GT should already be unit; tiny drift is fine).
        n = np.linalg.norm(q, axis=1, keepdims=True)
        return q / np.clip(n, 1e-9, None)

    if "physics_gt.ee_orientation" in df.columns:
        out["ee_orientation"] = _canon_q(_stack(df, "physics_gt.ee_orientation"))[sel]
    if "physics_gt.ee_angular_velocity" in df.columns:
        out["ee_angular_velocity"] = _stack(df, "physics_gt.ee_angular_velocity")[sel]
    # Object rotational (object/peg/nut/handle — handle has no orientation)
    if obj_keys is not None:
        pref = load_tasks()[task]["object_prefix"]
        rot_col = f"physics_gt.{pref}_orientation"
        ang_col = f"physics_gt.{pref}_angular_velocity"
        if rot_col in df.columns:
            out["obj_orientation"] = _canon_q(_stack(df, rot_col))[sel]
        if ang_col in df.columns:
            out["obj_angular_velocity"] = _stack(df, ang_col)[sel]

    # Tier-Contact: contact_flag, contact_force (3D), contact_force_mag (1D), contact_point (3D)
    # Available for push/strike/drawer/peg_insert/nut_thread (not reach)
    if "physics_gt.contact_flag" in df.columns:
        out["contact_flag"] = _stack(df, "physics_gt.contact_flag")[sel].astype(np.float32)
    if "physics_gt.contact_force" in df.columns:
        cf = _stack(df, "physics_gt.contact_force")[sel]
        out["contact_force"] = cf
        mag = np.linalg.norm(cf, axis=1, keepdims=False).astype(np.float32)
        out["contact_force_mag"] = mag
        # Heavy-tailed magnitudes get dominated by impact spikes (per Codex);
        # add log1p version as a more stable sibling target.
        out["contact_force_log1p_mag"] = np.log1p(mag).astype(np.float32)
    if "physics_gt.contact_point" in df.columns:
        out["contact_point"] = _stack(df, "physics_gt.contact_point")[sel]

    # Tier-Progress: phase + per-task task-progress scalars
    if "physics_gt.phase" in df.columns:
        out["phase"] = _stack(df, "physics_gt.phase")[sel].astype(np.float32)

    # Per-task progress / distance scalars (variable names differ)
    progress_cols = {
        "push":       ["physics_gt.ee_to_object_distance", "physics_gt.object_to_target_distance"],
        "strike":     ["physics_gt.ee_to_object_distance", "physics_gt.object_to_target_distance",
                       "physics_gt.ball_planar_travel_distance"],
        "reach":      ["physics_gt.ee_to_target_distance"],
        "drawer":     ["physics_gt.drawer_joint_pos", "physics_gt.drawer_joint_vel",
                       "physics_gt.drawer_opening_extent"],
        "peg_insert": ["physics_gt.insertion_depth", "physics_gt.peg_hole_lateral_error"],
        "nut_thread": ["physics_gt.axial_progress", "physics_gt.nut_bolt_relative_angle"],
    }
    for col in progress_cols.get(task, []):
        if col in df.columns:
            short = col.replace("physics_gt.", "")
            out[short] = _stack(df, col)[sel].astype(np.float32)

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


def task_target_keys(task: str, tier: str = "kinematic") -> list[str]:
    """Per-task target list. Tiers:
       'kinematic'  : original spec § 4 — pos, vel, speed, direction, accel, accel_mag (ee + obj)
       'rotational' : ee_orientation, ee_angular_velocity, obj_orientation, obj_angular_velocity
       'contact'    : contact_flag, contact_force, contact_force_mag, contact_point
       'progress'   : phase + per-task progress scalars (insertion_depth, drawer_joint_pos, etc.)
       'all_extended': contact + rotational + progress (kinematic excluded — already done)
    """
    cfg = load_tasks()[task]
    if tier == "kinematic":
        keys = ["ee_position", "ee_velocity", "ee_speed", "ee_direction", "ee_acceleration", "ee_accel_mag"]
        if cfg["has_object"]:
            keys += ["obj_position", "obj_velocity", "obj_speed", "obj_direction", "obj_acceleration", "obj_accel_mag"]
        return keys

    if tier == "rotational":
        keys = ["ee_orientation", "ee_angular_velocity"]
        # obj_orientation only present for tasks with object that has orientation (push, strike, peg, nut)
        if task in ("push", "strike", "peg_insert", "nut_thread"):
            keys += ["obj_orientation", "obj_angular_velocity"]
        return keys

    if tier == "contact":
        # Reach has no contact info
        if task == "reach":
            return []
        return ["contact_flag", "contact_force", "contact_force_mag",
                "contact_force_log1p_mag", "contact_point"]

    if tier == "progress":
        # phase + task-specific progress scalars (column names without 'physics_gt.' prefix)
        per_task = {
            "push":       ["ee_to_object_distance", "object_to_target_distance"],
            "strike":     ["ee_to_object_distance", "object_to_target_distance", "ball_planar_travel_distance"],
            "reach":      ["ee_to_target_distance"],
            "drawer":     ["drawer_joint_pos", "drawer_joint_vel", "drawer_opening_extent"],
            "peg_insert": ["insertion_depth", "peg_hole_lateral_error"],
            "nut_thread": ["axial_progress", "nut_bolt_relative_angle"],
        }
        return ["phase"] + per_task.get(task, [])

    if tier == "all_extended":
        return (task_target_keys(task, "rotational") +
                task_target_keys(task, "contact") +
                task_target_keys(task, "progress"))

    raise ValueError(f"Unknown target tier: {tier}")
