"""Automated dataset verification for PhysREPA Step 0 data.

Usage:
    /isaac-sim/python.sh verify_dataset.py --task push --data_dir /path/to/push --level 0
    /isaac-sim/python.sh verify_dataset.py --task push --data_dir /path/to/push --level all
"""

import argparse
import json
import os

import numpy as np
import pandas as pd


# Expected physics param keys per task
TASK_PARAM_KEYS = {
    "push": ["object_0_mass", "object_0_static_friction", "surface_static_friction"],
    "strike": ["object_0_mass", "object_0_static_friction", "surface_static_friction", "object_0_restitution"],
    "peg_insert": ["peg_static_friction", "peg_mass", "hole_static_friction"],
    "nut_thread": ["nut_static_friction", "nut_mass", "bolt_static_friction"],
    "drawer": ["drawer_joint_damping", "handle_static_friction", "drawer_handle_mass"],
    "reach": [],
}


def load_episodes_meta(data_dir):
    path = os.path.join(data_dir, "meta", "episodes.jsonl")
    with open(path) as f:
        return [json.loads(line) for line in f]


def load_info(data_dir):
    with open(os.path.join(data_dir, "meta", "info.json")) as f:
        return json.load(f)


def load_parquet(data_dir, ep_idx):
    chunk_idx = ep_idx // 1000
    path = os.path.join(data_dir, "data", f"chunk-{chunk_idx:03d}", f"episode_{ep_idx:06d}.parquet")
    return pd.read_parquet(path)


# ============================================================
# Level 0: Physics param randomization
# ============================================================

def verify_level0(data_dir, task_name):
    """Check that physics params vary across episodes."""
    print("\n=== Level 0: Physics Param Randomization ===")
    episodes = load_episodes_meta(data_dir)
    param_keys = TASK_PARAM_KEYS.get(task_name, [])

    if not param_keys:
        print(f"  {task_name}: No physics params (negative control). SKIP.")
        return True

    all_pass = True
    for key in param_keys:
        values = [ep.get(key) for ep in episodes if ep.get(key) is not None]
        if not values:
            print(f"  {key}: NOT FOUND in episodes.jsonl ❌")
            all_pass = False
            continue

        arr = np.array(values)
        unique = len(set(values))
        print(f"  {key}: min={arr.min():.4f}, max={arr.max():.4f}, "
              f"mean={arr.mean():.4f}, std={arr.std():.4f}, unique={unique}/{len(values)}", end="")

        if unique <= 1:
            print(" ❌ FAIL (no variation)")
            all_pass = False
        elif arr.std() < 1e-6:
            print(" ❌ FAIL (zero variance)")
            all_pass = False
        else:
            print(" ✅ PASS")

    return all_pass


# ============================================================
# Level 1: GT consistency
# ============================================================

def verify_level1(data_dir, task_name, sample_size=10):
    """Verify velocity ~ delta_pos/dt, contact consistency."""
    print("\n=== Level 1: GT Consistency ===")
    episodes = load_episodes_meta(data_dir)
    info = load_info(data_dir)
    fps = info.get("fps", 50)
    dt = 1.0 / fps

    sample_indices = np.random.choice(len(episodes), min(sample_size, len(episodes)), replace=False)
    all_pass = True

    # 1-1: Velocity ≈ Δposition / Δt
    print("\n  --- 1-1: Velocity vs finite difference ---")
    vel_corrs = []
    for idx in sample_indices:
        df = load_parquet(data_dir, idx)
        if "physics_gt.ee_position" not in df.columns or "physics_gt.ee_velocity" not in df.columns:
            continue
        pos = np.array(df["physics_gt.ee_position"].tolist())
        vel = np.array(df["physics_gt.ee_velocity"].tolist())
        if len(pos) < 3:
            continue
        computed_vel = np.diff(pos, axis=0) / dt
        # Correlation between computed and recorded velocity
        for dim in range(3):
            corr = np.corrcoef(computed_vel[:, dim], vel[1:, dim])[0, 1]
            vel_corrs.append(corr)

    if vel_corrs:
        mean_corr = np.nanmean(vel_corrs)
        print(f"  ee velocity-position correlation: {mean_corr:.4f}", end="")
        if mean_corr > 0.8:
            print(" ✅ PASS")
        elif mean_corr > 0.5:
            print(" ⚠️ WARNING (low correlation)")
        else:
            print(" ❌ FAIL")
            all_pass = False
    else:
        print("  ee velocity: no data to check. SKIP.")

    # 1-2: Contact flag ↔ force magnitude
    print("\n  --- 1-2: Contact flag vs force magnitude ---")
    flag_force_ok = 0
    flag_force_total = 0
    for idx in sample_indices:
        df = load_parquet(data_dir, idx)
        if "physics_gt.contact_flag" not in df.columns or "physics_gt.contact_force" not in df.columns:
            continue
        flags = np.array(df["physics_gt.contact_flag"].tolist()).flatten()
        forces = np.array(df["physics_gt.contact_force"].tolist())
        force_mags = np.linalg.norm(forces, axis=-1)
        # Check consistency
        for t in range(len(flags)):
            flag_force_total += 1
            if flags[t] > 0.5 and force_mags[t] > 0.1:
                flag_force_ok += 1
            elif flags[t] < 0.5 and force_mags[t] < 1.0:
                flag_force_ok += 1
            # else: mismatch

    if flag_force_total > 0:
        ratio = flag_force_ok / flag_force_total
        print(f"  Contact flag-force consistency: {ratio:.4f} ({flag_force_ok}/{flag_force_total})", end="")
        if ratio > 0.9:
            print(" ✅ PASS")
        elif ratio > 0.7:
            print(" ⚠️ WARNING")
        else:
            print(" ❌ FAIL")
            all_pass = False
    else:
        print("  Contact: no data. SKIP.")

    return all_pass


# ============================================================
# Level 2: Metadata consistency
# ============================================================

def verify_level2(data_dir, task_name):
    """Check info.json ↔ episodes.jsonl ↔ parquet consistency."""
    print("\n=== Level 2: Metadata Consistency ===")
    info = load_info(data_dir)
    episodes = load_episodes_meta(data_dir)
    all_pass = True

    # 2-1: Episode count
    print(f"  info.total_episodes={info.get('total_episodes')}, episodes.jsonl={len(episodes)}", end="")
    if info.get("total_episodes") == len(episodes):
        print(" ✅")
    else:
        print(" ❌ MISMATCH")
        all_pass = False

    # 2-2: Total frames
    sum_lengths = sum(ep["length"] for ep in episodes)
    print(f"  info.total_frames={info.get('total_frames')}, sum(lengths)={sum_lengths}", end="")
    if info.get("total_frames") == sum_lengths:
        print(" ✅")
    else:
        print(" ❌ MISMATCH")
        all_pass = False

    # 2-3: Parquet row counts (sample)
    print("  Checking parquet row counts (sample 10)...")
    sample = episodes[:10]
    for ep in sample:
        try:
            df = load_parquet(data_dir, ep["episode_index"])
            if len(df) != ep["length"]:
                print(f"    ep {ep['episode_index']}: parquet={len(df)}, meta={ep['length']} ❌")
                all_pass = False
        except FileNotFoundError:
            print(f"    ep {ep['episode_index']}: parquet file NOT FOUND ❌")
            all_pass = False

    # 2-4: Success flags
    success_count = sum(1 for ep in episodes if ep.get("success", False))
    print(f"  Success: {success_count}/{len(episodes)} ({success_count/len(episodes)*100:.1f}%)")

    return all_pass


# ============================================================
# Level 3: Video quality
# ============================================================

def verify_level3(data_dir, task_name, sample_size=5):
    """Check video resolution, frame count, not all-black."""
    print("\n=== Level 3: Video Quality ===")
    info = load_info(data_dir)
    episodes = load_episodes_meta(data_dir)
    all_pass = True

    try:
        import imageio
    except ImportError:
        print("  imageio not available. SKIP.")
        return True

    sample_indices = list(range(min(sample_size, len(episodes))))

    for cam_key in ["observation.images.image_0", "observation.images.image_1"]:
        print(f"\n  --- {cam_key} ---")
        for idx in sample_indices:
            ep = episodes[idx]
            chunk_idx = idx // 1000
            video_path = os.path.join(data_dir, "videos", f"chunk-{chunk_idx:03d}", cam_key, f"episode_{idx:06d}.mp4")
            if not os.path.exists(video_path):
                print(f"    ep {idx}: video NOT FOUND ❌")
                all_pass = False
                continue

            try:
                reader = imageio.get_reader(video_path)
                meta = reader.get_meta_data()
                n_frames = reader.count_frames()
                first_frame = reader.get_data(0)
                h, w = first_frame.shape[:2]

                # Resolution check
                res_ok = h == 384 and w == 384
                # Frame count check
                frame_ok = n_frames == ep["length"]
                # Not all-black
                mean_px = first_frame.mean()
                black_ok = mean_px > 5.0

                status = "✅" if (res_ok and frame_ok and black_ok) else "❌"
                issues = []
                if not res_ok:
                    issues.append(f"res={h}x{w}")
                if not frame_ok:
                    issues.append(f"frames={n_frames}vs{ep['length']}")
                if not black_ok:
                    issues.append("all-black")

                if issues:
                    print(f"    ep {idx}: {status} {', '.join(issues)}")
                    all_pass = False
                reader.close()
            except Exception as e:
                print(f"    ep {idx}: ERROR {e} ❌")
                all_pass = False

    return all_pass


# ============================================================
# Level 4: Physics-dynamics correlation
# ============================================================

def verify_level4(data_dir, task_name, max_episodes=200):
    """Check physics params correlate with observed dynamics."""
    print("\n=== Level 4: Physics-Dynamics Correlation ===")
    episodes = load_episodes_meta(data_dir)
    all_pass = True

    n = min(max_episodes, len(episodes))
    if task_name not in ("push", "strike", "drawer"):
        print(f"  {task_name}: correlation check not implemented. SKIP.")
        return True

    frictions = []
    displacements = []

    for idx in range(n):
        ep = episodes[idx]
        try:
            df = load_parquet(data_dir, idx)
        except Exception:
            continue

        if task_name in ("push", "strike"):
            friction = ep.get("object_0_static_friction", None)
            if friction is None:
                continue
            obj_pos = np.array(df["physics_gt.object_position"].tolist())
            disp = np.linalg.norm(obj_pos[-1, :2] - obj_pos[0, :2])
            frictions.append(friction)
            displacements.append(disp)
        elif task_name == "drawer":
            damping = ep.get("drawer_joint_damping", None)
            if damping is None:
                continue
            if "physics_gt.drawer_joint_pos" in df.columns:
                jpos = np.array(df["physics_gt.drawer_joint_pos"].tolist())
                final_open = float(jpos[-1]) if jpos.size > 0 else 0.0
                frictions.append(damping)
                displacements.append(final_open)

    if len(frictions) < 10:
        print(f"  Not enough data ({len(frictions)} episodes). SKIP.")
        return True

    from scipy.stats import spearmanr
    corr, pval = spearmanr(frictions, displacements)

    param_name = "friction" if task_name in ("push", "strike") else "damping"
    dynamics_name = "displacement" if task_name in ("push", "strike") else "opening"

    print(f"  {param_name} vs {dynamics_name}: corr={corr:.3f}, p={pval:.4f}", end="")

    if task_name in ("push", "strike"):
        # Higher friction → less displacement (expect negative correlation)
        if corr < -0.1 and pval < 0.05:
            print(" ✅ PASS (negative correlation)")
        elif abs(corr) < 0.1:
            print(" ⚠️ WARNING (weak correlation)")
            all_pass = False
        else:
            print(f" ❌ FAIL (expected negative, got {corr:.3f})")
            all_pass = False
    elif task_name == "drawer":
        # Higher damping → less opening (expect negative correlation)
        if corr < -0.1 and pval < 0.05:
            print(" ✅ PASS (negative correlation)")
        elif abs(corr) < 0.1:
            print(" ⚠️ WARNING (weak correlation)")
        else:
            print(f" ❌ unexpected direction: {corr:.3f}")

    return all_pass


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Verify PhysREPA dataset")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--level", type=str, default="all", help="0,1,2,3,4 or all")
    args = parser.parse_args()

    levels = [0, 1, 2, 3, 4] if args.level == "all" else [int(x) for x in args.level.split(",")]

    print(f"Verifying {args.task} at {args.data_dir}")
    print(f"Levels: {levels}")

    results = {}
    if 0 in levels:
        results["L0"] = verify_level0(args.data_dir, args.task)
    if 1 in levels:
        results["L1"] = verify_level1(args.data_dir, args.task)
    if 2 in levels:
        results["L2"] = verify_level2(args.data_dir, args.task)
    if 3 in levels:
        results["L3"] = verify_level3(args.data_dir, args.task)
    if 4 in levels:
        results["L4"] = verify_level4(args.data_dir, args.task)

    print(f"\n{'='*50}")
    print("VERIFICATION SUMMARY")
    for level, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {level}: {status}")

    all_pass = all(results.values())
    print(f"\nOverall: {'✅ ALL PASS' if all_pass else '❌ SOME FAILED'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    exit(main())
