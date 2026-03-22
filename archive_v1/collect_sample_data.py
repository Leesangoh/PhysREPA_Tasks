"""Collect sample data in LeRobot V2 format (identical to BridgeData).

Runs each of the 4 PhysREPA tasks with random actions for 5 episodes each,
saving in the exact same format as BridgeData V2.

Usage:
    cd /home/solee/IsaacLab
    PYTHONPATH=/home/solee:$PYTHONPATH ./isaaclab.sh -p \
        /home/solee/physrepa_tasks/collect_sample_data.py --task lift --num_episodes 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import functools

print = functools.partial(print, flush=True)

import torch
import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect PhysREPA sample data in LeRobot V2 format")
parser.add_argument("--task", type=str, required=True, choices=["lift", "pick_place", "push", "stack"])
parser.add_argument("--num_episodes", type=int, default=5)
parser.add_argument("--use_oracle", action="store_true", help="Use scripted oracle policy instead of random actions")
parser.add_argument("--output_dir", type=str, default="/mnt/md1/solee/data/isaac_physrepa")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Post-launch imports
import pandas as pd
import imageio

from isaaclab.envs import ManagerBasedRLEnv

from physrepa_tasks.envs.lift_env_cfg import PhysREPALiftEnvCfg
from physrepa_tasks.envs.pick_place_env_cfg import PhysREPAPickPlaceEnvCfg
from physrepa_tasks.envs.push_env_cfg import PhysREPAPushEnvCfg
from physrepa_tasks.envs.stack_env_cfg import PhysREPAStackEnvCfg
from physrepa_tasks.policies.scripted_policy import LiftPolicy, PickPlacePolicy, PushPolicy, StackPolicy

# Phase labels for future use (all random actions = idle)
PHASES = {
    0: "reach",
    1: "grasp",
    2: "lift",
    3: "transport",
    4: "place",
    5: "release",
    6: "retract",
    7: "idle",
}

# Fixed color instructions (no color randomization)
TASK_INSTRUCTION_TEMPLATES = {
    "lift": "Lift the red cube",
    "pick_place": "Pick up the red cube and place it on the green marker",
    "push": "Push the red cube to the orange marker",
    "stack": "Stack the small red cube on top of the large blue cube",
}

TASK_CONFIGS = {
    "lift": PhysREPALiftEnvCfg,
    "pick_place": PhysREPAPickPlaceEnvCfg,
    "push": PhysREPAPushEnvCfg,
    "stack": PhysREPAStackEnvCfg,
}

TASK_POLICIES = {
    "lift": LiftPolicy,
    "pick_place": PickPlacePolicy,
    "push": PushPolicy,
    "stack": StackPolicy,
}

# Map policy state index → phase label index for each task
TASK_STATE_TO_PHASE = {
    "lift": {0: 0, 1: 0, 2: 1, 3: 2, 4: 7},  # approach→reach, descend→reach, grasp, lift, hold→idle
    "pick_place": {0: 0, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6},  # reach, reach, grasp, lift, transport, place, release, retract
    "push": {0: 0, 1: 0, 2: 5, 3: 7},  # approach→reach, descend→reach, push(=5), done→idle
    "stack": {0: 0, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6},  # reach, reach, grasp, lift, transport, place, release, retract
}

# Object size constants for on_surface checks
TABLE_SURFACE_Z = 0.0
CUBE_HALF_SIZE = 0.03  # 0.06m cube / 2 (lift, pick_place, push)
CUBE_A_HALF_SIZE = 0.025  # 0.05m small cube / 2 (stack)
CUBE_B_HALF_SIZE = 0.035  # 0.07m large cube / 2 (stack)
ON_SURFACE_MARGIN = 0.01

FFMPEG_PATH = "/isaac-sim/kit/python/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
# FPS is derived from config in collect_task(), not hardcoded here
VIDEO_CODEC = "libx264"  # h264, widely compatible; can re-encode to av1 later
CHUNKS_SIZE = 1000


def encode_video(frames: list[np.ndarray], output_path: str, fps: int = 50):
    """Encode list of RGB numpy arrays to MP4 using imageio-ffmpeg."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = imageio.get_writer(
        output_path,
        fps=fps,
        codec=VIDEO_CODEC,
        quality=8,  # good quality
        pixelformat="yuv420p",
        ffmpeg_params=["-preset", "fast"],
    )
    for frame in frames:
        writer.append_data(frame)
    writer.close()


def ee_state_to_8d(ee_pos_b: np.ndarray, ee_quat_b: np.ndarray, gripper_pos: float) -> np.ndarray:
    """Convert EE position + quaternion (both in base frame) to BridgeData-compatible 8D state.

    BridgeData state: [x, y, z, roll, pitch, yaw, pad, gripper]
    Args:
        ee_pos_b: (3,) EE position in robot base frame.
        ee_quat_b: (4,) EE orientation in robot base frame (w, x, y, z).
        gripper_pos: Gripper opening normalized to 0-1.
    """
    from scipy.spatial.transform import Rotation

    # scipy expects [x, y, z, w]
    quat_xyzw = np.array([ee_quat_b[1], ee_quat_b[2], ee_quat_b[3], ee_quat_b[0]])
    rpy = Rotation.from_quat(quat_xyzw).as_euler("xyz")
    return np.array([ee_pos_b[0], ee_pos_b[1], ee_pos_b[2], rpy[0], rpy[1], rpy[2], 0.0, gripper_pos], dtype=np.float32)


def get_ee_pose_in_base_frame(env):
    """Get EE position and orientation in robot base frame.

    Returns:
        ee_pos_b: (3,) numpy array, position in base frame.
        ee_quat_b: (4,) numpy array, orientation (w,x,y,z) in base frame.
    """
    from isaaclab.utils.math import subtract_frame_transforms

    robot = env.scene["robot"]
    ee_frame = env.scene["ee_frame"]

    ee_pos_w = ee_frame.data.target_pos_w[0, 0, :]
    ee_quat_w = ee_frame.data.target_quat_w[0, 0, :]
    root_pos_w = robot.data.root_pos_w[0]
    root_quat_w = robot.data.root_quat_w[0]

    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        root_pos_w.unsqueeze(0), root_quat_w.unsqueeze(0),
        ee_pos_w.unsqueeze(0), ee_quat_w.unsqueeze(0),
    )
    return ee_pos_b[0].cpu().numpy(), ee_quat_b[0].cpu().numpy()


def get_task_instruction(task_name: str, env) -> str:
    """Build language instruction from task name. Colors are fixed (no randomization)."""
    return TASK_INSTRUCTION_TEMPLATES[task_name]


def read_physics_params(env, task_name: str) -> dict:
    """Read actual randomized physics parameters from the simulation after reset."""
    params = {}

    if task_name == "stack":
        # Cube A (object_0)
        cube_a = env.scene["cube_a"]
        mass_a = cube_a.root_physx_view.get_masses()[0, 0].item()
        mat_a = cube_a.root_physx_view.get_material_properties()
        params["object_0_mass"] = round(mass_a, 4)
        params["object_0_static_friction"] = round(mat_a[0, 0, 0].item(), 4)
        params["object_0_dynamic_friction"] = round(mat_a[0, 0, 1].item(), 4)
        params["object_0_color"] = "red"
        params["object_0_type"] = "cube"

        # Cube B (object_1)
        cube_b = env.scene["cube_b"]
        mass_b = cube_b.root_physx_view.get_masses()[0, 0].item()
        mat_b = cube_b.root_physx_view.get_material_properties()
        params["object_1_mass"] = round(mass_b, 4)
        params["object_1_static_friction"] = round(mat_b[0, 0, 0].item(), 4)
        params["object_1_dynamic_friction"] = round(mat_b[0, 0, 1].item(), 4)
        params["object_1_color"] = "blue"
        params["object_1_type"] = "cube"
    else:
        obj = env.scene["object"]
        mass = obj.root_physx_view.get_masses()[0, 0].item()
        mat_props = obj.root_physx_view.get_material_properties()
        params["object_0_mass"] = round(mass, 4)
        params["object_0_static_friction"] = round(mat_props[0, 0, 0].item(), 4)
        params["object_0_dynamic_friction"] = round(mat_props[0, 0, 1].item(), 4)
        params["object_0_color"] = "red"
        params["object_0_type"] = "cube"

    if task_name == "push":
        surface = env.scene["surface"]
        surface_mat = surface.root_physx_view.get_material_properties()
        params["surface_static_friction"] = round(surface_mat[0, 0, 0].item(), 4)
        params["surface_dynamic_friction"] = round(surface_mat[0, 0, 1].item(), 4)

    return params


def collect_task(task_name: str, num_episodes: int, output_dir: str, use_oracle: bool = False):
    """Collect episodes for a single task and save in LeRobot V2 format."""
    cfg_cls = TASK_CONFIGS[task_name]
    cfg = cfg_cls()
    cfg.scene.num_envs = 1  # collect one env at a time for clean episodes

    env = ManagerBasedRLEnv(cfg=cfg)
    print(f"\n{'='*60}")
    print(f"Collecting {num_episodes} episodes for task: {task_name}")
    print(f"{'='*60}")

    robot = env.scene["robot"]
    finger_ids = robot.find_joints("panda_finger.*")[0]

    # Get EE frame sensor and its body
    ee_frame = env.scene["ee_frame"]

    # Derive FPS and dt from config (Critical fix #5: no hardcoding)
    dt = cfg.sim.dt * cfg.decimation
    fps = int(1.0 / dt)

    # Oracle policy
    oracle_policy = None
    if use_oracle:
        policy_cls = TASK_POLICIES[task_name]
        oracle_policy = policy_cls(num_envs=1, device=env.device)
        print(f"  Using oracle policy: {policy_cls.__name__}")

    episodes_meta = []
    tasks_meta_dict = {}  # task_description -> task_index
    tasks_meta_list = []
    global_index = 0
    all_episode_stats = {"observation.state": [], "action": [], "timestamp": []}

    # Physics GT column names (base set for all tasks)
    physics_gt_keys = [
        "physics_gt.ee_position",
        "physics_gt.ee_velocity",
        "physics_gt.ee_acceleration",
        "physics_gt.object_position",
        "physics_gt.object_orientation",
        "physics_gt.object_velocity",
        "physics_gt.object_angular_velocity",
        "physics_gt.object_acceleration",
        "physics_gt.ee_to_object_distance",
        "physics_gt.contact_flag",
        "physics_gt.contact_force",
        "physics_gt.object_on_surface",
        "physics_gt.contact_point",
        "physics_gt.phase",
    ]

    # Add target_position for pick_place and push (Critical fix #2)
    if task_name in ("pick_place", "push"):
        physics_gt_keys.append("physics_gt.target_position")

    # Additional keys for stack task
    stack_extra_keys = [
        "physics_gt.object_1_position",
        "physics_gt.object_1_orientation",
        "physics_gt.object_1_velocity",
        "physics_gt.object_1_angular_velocity",
        "physics_gt.object_1_acceleration",
        "physics_gt.ee_to_object_1_distance",
        "physics_gt.object_0_to_object_1_distance",
        "physics_gt.object_1_on_surface",
        "physics_gt.object_object_contact_flag",
        "physics_gt.object_object_contact_force",
        "physics_gt.object_object_contact_point",
    ]

    if task_name == "stack":
        physics_gt_keys = physics_gt_keys + stack_extra_keys

    all_episodes_physics_gt = []

    for ep_idx in range(num_episodes):
        print(f"\n--- Episode {ep_idx} ---")

        # Reset with overlap check for pick_place/push
        # Ensure object and target are at least 0.1m apart
        for _reset_attempt in range(10):
            obs, info = env.reset()
            if task_name in ("pick_place", "push"):
                gt = obs["physics_gt"]
                obj_pos = gt["object_position"][0, :2].cpu()
                target_pos = gt["target_position"][0, :2].cpu()
                dist = torch.norm(obj_pos - target_pos).item()
                if dist >= 0.10:
                    break
                print(f"  Re-reset: object-target too close ({dist:.3f}m)")
            else:
                break

        # Get task instruction (fixed colors, no randomization)
        task_description = get_task_instruction(task_name, env)
        if task_description not in tasks_meta_dict:
            task_idx = len(tasks_meta_dict)
            tasks_meta_dict[task_description] = task_idx
            tasks_meta_list.append({"task_index": task_idx, "task": task_description})
        current_task_idx = tasks_meta_dict[task_description]
        print(f"  Instruction: {task_description}")

        # Read physics params after reset (when randomization has been applied)
        physics_params = read_physics_params(env, task_name)
        print(f"  Physics params: {physics_params}")

        # Set adaptive push parameters based on friction
        if task_name == "push" and oracle_policy is not None and hasattr(oracle_policy, 'set_friction'):
            sf = physics_params.get('surface_static_friction', 0.5)
            of = physics_params.get('object_0_static_friction', 0.5)
            oracle_policy.set_friction(sf, of)
            print(f"  Push adaptive: surf_f={sf:.3f}, obj_f={of:.3f}")

        # Storage for this episode
        ep_states = []
        ep_actions = []
        ep_timestamps = []
        ep_physics_gt = {k: [] for k in physics_gt_keys}
        ep_frames_table = []
        ep_frames_wrist = []
        ep_rewards = []

        # Sync target marker to command target position (pick_place/push)
        if task_name in ("pick_place", "push"):
            from physrepa_tasks.mdp.sync_marker import sync_target_marker
            sync_target_marker(env, env_ids=None, command_name="object_pose")

        # Warmup: 0.5 seconds of no-action steps for rendering stabilization
        warmup_steps = int(0.5 / dt)
        for _ in range(warmup_steps):
            obs, _, _, _, _ = env.step(torch.zeros(1, 7, device=env.device))

        # max_steps = remaining episode time AFTER warmup
        max_steps = int(cfg.episode_length_s / dt) - warmup_steps

        # Reset oracle policy after warmup
        if oracle_policy is not None:
            oracle_policy.reset()

        # Previous velocities for acceleration computation
        prev_ee_vel = None
        prev_obj_vel = None
        prev_obj1_vel = None  # for stack task cube_b

        for step in range(max_steps):
            # Generate action
            if oracle_policy is not None:
                action = oracle_policy.get_action(obs)
            else:
                action = torch.zeros(1, 7, device=env.device)
                action[0, :6] = torch.randn(6, device=env.device) * 0.1
                action[0, 6] = 1.0 if step < max_steps // 2 else -1.0

            # Record pre-step data
            gt = obs["physics_gt"]

            # EE state: convert to BridgeData 8D format [x,y,z,roll,pitch,yaw,pad,gripper]
            # Critical fix #1: both position AND orientation in base frame
            ee_pos_b_np, ee_quat_b_np = get_ee_pose_in_base_frame(env)
            gripper_val = robot.data.joint_pos[0, finger_ids].mean().item() / 0.04  # normalize to 0-1
            state_8d = ee_state_to_8d(ee_pos_b_np, ee_quat_b_np, gripper_val)
            ep_states.append(state_8d)

            # Action: 7D [dx, dy, dz, droll, dpitch, dyaw, gripper]
            action_7d = action[0].cpu().numpy().astype(np.float32)
            # Normalize gripper to 0-1 range (currently -1/+1)
            action_7d[6] = 0.0 if action_7d[6] < 0 else 1.0
            ep_actions.append(action_7d)

            # Timestamp
            ep_timestamps.append(np.float32(step / fps))

            # Physics GT: EE
            ee_pos_np = gt["ee_position"][0].cpu().numpy()
            ee_vel_np = gt["ee_velocity"][0].cpu().numpy()
            ep_physics_gt["physics_gt.ee_position"].append(ee_pos_np)
            ep_physics_gt["physics_gt.ee_velocity"].append(ee_vel_np)

            # EE acceleration via finite difference
            if prev_ee_vel is None:
                prev_ee_vel = ee_vel_np.copy()
            ee_accel = (ee_vel_np - prev_ee_vel) / dt
            ep_physics_gt["physics_gt.ee_acceleration"].append(ee_accel.astype(np.float32))
            prev_ee_vel = ee_vel_np.copy()

            # Handle different observation keys for different tasks
            if "object_position" in gt:
                obj_pos_np = gt["object_position"][0].cpu().numpy()
                obj_vel_np = gt["object_velocity"][0].cpu().numpy()
                ep_physics_gt["physics_gt.object_position"].append(obj_pos_np)
                ep_physics_gt["physics_gt.object_orientation"].append(gt["object_orientation"][0].cpu().numpy())
                ep_physics_gt["physics_gt.object_velocity"].append(obj_vel_np)
                ep_physics_gt["physics_gt.object_angular_velocity"].append(gt["object_angular_velocity"][0].cpu().numpy())
                ep_physics_gt["physics_gt.ee_to_object_distance"].append(gt["ee_to_object_distance"][0].cpu().numpy())

                # Object acceleration
                if prev_obj_vel is None:
                    prev_obj_vel = obj_vel_np.copy()
                obj_accel = (obj_vel_np - prev_obj_vel) / dt
                ep_physics_gt["physics_gt.object_acceleration"].append(obj_accel.astype(np.float32))
                prev_obj_vel = obj_vel_np.copy()

                # Object on surface check
                obj_z = obj_pos_np[2]
                on_surface = 1.0 if obj_z < (TABLE_SURFACE_Z + CUBE_HALF_SIZE + ON_SURFACE_MARGIN) else 0.0
                ep_physics_gt["physics_gt.object_on_surface"].append(np.array([on_surface], dtype=np.float32))

            elif "cube_a_position" in gt:
                # Stack task: cube_a = object_0
                cube_a_pos_np = gt["cube_a_position"][0].cpu().numpy()
                cube_a_vel_np = gt["cube_a_velocity"][0].cpu().numpy()
                ep_physics_gt["physics_gt.object_position"].append(cube_a_pos_np)
                ep_physics_gt["physics_gt.object_orientation"].append(gt["cube_a_orientation"][0].cpu().numpy())
                ep_physics_gt["physics_gt.object_velocity"].append(cube_a_vel_np)
                ep_physics_gt["physics_gt.object_angular_velocity"].append(gt["cube_a_angular_velocity"][0].cpu().numpy())
                ep_physics_gt["physics_gt.ee_to_object_distance"].append(gt["ee_to_cube_a_distance"][0].cpu().numpy())

                # Cube A acceleration
                if prev_obj_vel is None:
                    prev_obj_vel = cube_a_vel_np.copy()
                obj_accel = (cube_a_vel_np - prev_obj_vel) / dt
                ep_physics_gt["physics_gt.object_acceleration"].append(obj_accel.astype(np.float32))
                prev_obj_vel = cube_a_vel_np.copy()

                # Cube A on surface
                cube_a_z = cube_a_pos_np[2]
                on_surface_a = 1.0 if cube_a_z < (TABLE_SURFACE_Z + CUBE_A_HALF_SIZE + ON_SURFACE_MARGIN) else 0.0
                ep_physics_gt["physics_gt.object_on_surface"].append(np.array([on_surface_a], dtype=np.float32))

                # Cube B = object_1
                cube_b_pos_np = gt["cube_b_position"][0].cpu().numpy()
                cube_b_vel_np = gt["cube_b_velocity"][0].cpu().numpy()
                ep_physics_gt["physics_gt.object_1_position"].append(cube_b_pos_np)
                ep_physics_gt["physics_gt.object_1_orientation"].append(gt["cube_b_orientation"][0].cpu().numpy())
                ep_physics_gt["physics_gt.object_1_velocity"].append(cube_b_vel_np)
                ep_physics_gt["physics_gt.object_1_angular_velocity"].append(gt["cube_b_angular_velocity"][0].cpu().numpy())

                # Cube B acceleration
                if prev_obj1_vel is None:
                    prev_obj1_vel = cube_b_vel_np.copy()
                obj1_accel = (cube_b_vel_np - prev_obj1_vel) / dt
                ep_physics_gt["physics_gt.object_1_acceleration"].append(obj1_accel.astype(np.float32))
                prev_obj1_vel = cube_b_vel_np.copy()

                # EE to cube_b distance
                ee_to_b = gt["ee_to_cube_b_distance"][0].cpu().numpy()
                ep_physics_gt["physics_gt.ee_to_object_1_distance"].append(ee_to_b)

                # Cube A to Cube B distance
                a_to_b = gt["cube_a_to_cube_b_distance"][0].cpu().numpy()
                ep_physics_gt["physics_gt.object_0_to_object_1_distance"].append(a_to_b)

                # Cube B on surface
                cube_b_z = cube_b_pos_np[2]
                on_surface_b = 1.0 if cube_b_z < (TABLE_SURFACE_Z + CUBE_B_HALF_SIZE + ON_SURFACE_MARGIN) else 0.0
                ep_physics_gt["physics_gt.object_1_on_surface"].append(np.array([on_surface_b], dtype=np.float32))

                # Object-object contact (cube_a <-> cube_b)
                # Critical fix #3: use force_matrix_w (per-filter-body) not net_forces_w (total)
                cube_sensor = env.scene.sensors["cube_contact_sensor"]
                # force_matrix_w shape: (N, B, M, 3) where M=1 (only CubeB as filter)
                obj_obj_force = cube_sensor.data.force_matrix_w[0, 0, 0, :].cpu().numpy()
                obj_obj_flag = 1.0 if np.linalg.norm(obj_obj_force) > 0.5 else 0.0
                ep_physics_gt["physics_gt.object_object_contact_flag"].append(np.array([obj_obj_flag], dtype=np.float32))
                ep_physics_gt["physics_gt.object_object_contact_force"].append(obj_obj_force.astype(np.float32))

                # Object-object contact point
                obj_obj_contact_pos = cube_sensor.data.contact_pos_w[0, 0, 0, :].cpu().numpy()
                obj_obj_contact_pos = np.nan_to_num(obj_obj_contact_pos, nan=0.0)
                ep_physics_gt["physics_gt.object_object_contact_point"].append(obj_obj_contact_pos.astype(np.float32))

            # Gripper contact
            ep_physics_gt["physics_gt.contact_flag"].append(gt["contact_flag"][0].cpu().numpy())
            ep_physics_gt["physics_gt.contact_force"].append(gt["contact_force"][0].cpu().numpy())

            # Contact point from contact sensor
            sensor = env.scene.sensors["contact_sensor"]
            contact_pos = sensor.data.contact_pos_w[0, 0, 0, :].cpu().numpy()
            contact_pos = np.nan_to_num(contact_pos, nan=0.0)
            ep_physics_gt["physics_gt.contact_point"].append(contact_pos.astype(np.float32))

            # Phase label
            if oracle_policy is not None:
                state_idx = oracle_policy.state[0].item()
                phase_map = TASK_STATE_TO_PHASE.get(task_name, {})
                phase = phase_map.get(state_idx, 7)
            else:
                phase = 7  # idle for random actions
            ep_physics_gt["physics_gt.phase"].append(np.array([phase], dtype=np.float32))

            # Target position for pick_place/push (Critical fix #2)
            if task_name in ("pick_place", "push") and "target_position" in gt:
                target_3d = gt["target_position"][0, :3].cpu().numpy()
                ep_physics_gt["physics_gt.target_position"].append(target_3d.astype(np.float32))

            # Camera frames
            policy_obs = obs["policy"]
            if isinstance(policy_obs, dict):
                if "table_cam" in policy_obs:
                    ep_frames_table.append(policy_obs["table_cam"][0].cpu().numpy())
                if "wrist_cam" in policy_obs:
                    ep_frames_wrist.append(policy_obs["wrist_cam"][0].cpu().numpy())

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)

            # Record reward (Critical fix #4: for next.done)
            ep_rewards.append(reward[0].item())

            # Print progress
            if step % 100 == 0:
                print(f"  Step {step}/{max_steps}, reward={reward[0].item():.4f}")

        ep_length = len(ep_states)
        print(f"  Episode {ep_idx} complete: {ep_length} steps")

        # --- Save episode data ---
        chunk_idx = ep_idx // CHUNKS_SIZE
        chunk_dir = f"chunk-{chunk_idx:03d}"

        # 1. Save parquet
        parquet_dir = os.path.join(output_dir, "data", chunk_dir)
        os.makedirs(parquet_dir, exist_ok=True)

        df_data = {
            "observation.state": [s.tolist() for s in ep_states],
            "action": [a.tolist() for a in ep_actions],
            "timestamp": ep_timestamps,
            "frame_index": list(range(ep_length)),
            "episode_index": [ep_idx] * ep_length,
            "index": list(range(global_index, global_index + ep_length)),
            "task_index": [current_task_idx] * ep_length,
            "next.done": [False] * (ep_length - 1) + [True],
            "next.reward": ep_rewards[:ep_length],
        }

        # Add physics GT columns
        for gt_key in physics_gt_keys:
            df_data[gt_key] = [v.tolist() for v in ep_physics_gt[gt_key]]

        df = pd.DataFrame(df_data)
        parquet_path = os.path.join(parquet_dir, f"episode_{ep_idx:06d}.parquet")
        df.to_parquet(parquet_path, index=False)
        print(f"  Saved parquet: {parquet_path} ({os.path.getsize(parquet_path) / 1024:.1f} KB)")

        # 2. Save videos (MP4)
        if ep_frames_table:
            video_dir_table = os.path.join(output_dir, "videos", chunk_dir, "observation.images.image_0")
            video_path_table = os.path.join(video_dir_table, f"episode_{ep_idx:06d}.mp4")
            encode_video(ep_frames_table, video_path_table, fps=fps)
            print(f"  Saved table_cam video: {video_path_table} ({os.path.getsize(video_path_table) / 1024:.1f} KB)")

        if ep_frames_wrist:
            video_dir_wrist = os.path.join(output_dir, "videos", chunk_dir, "observation.images.image_1")
            video_path_wrist = os.path.join(video_dir_wrist, f"episode_{ep_idx:06d}.mp4")
            encode_video(ep_frames_wrist, video_path_wrist, fps=fps)
            print(f"  Saved wrist_cam video: {video_path_wrist} ({os.path.getsize(video_path_wrist) / 1024:.1f} KB)")

        # 3. Episode metadata (with physics params)
        ep_meta = {
            "episode_index": ep_idx,
            "tasks": [task_description],
            "length": ep_length,
            "success": False,  # random actions, no success
        }
        ep_meta.update(physics_params)
        episodes_meta.append(ep_meta)

        # Accumulate stats
        all_episode_stats["observation.state"].extend(ep_states)
        all_episode_stats["action"].extend(ep_actions)
        all_episode_stats["timestamp"].extend(ep_timestamps)
        all_episodes_physics_gt.append({k: list(v) for k, v in ep_physics_gt.items()})

        global_index += ep_length

    # --- Save metadata ---
    meta_dir = os.path.join(output_dir, "meta")
    os.makedirs(meta_dir, exist_ok=True)

    # info.json
    total_frames = global_index

    # Build physics GT shapes dict
    physics_gt_shapes = {
        "physics_gt.ee_position": [3],
        "physics_gt.ee_velocity": [3],
        "physics_gt.ee_acceleration": [3],
        "physics_gt.object_position": [3],
        "physics_gt.object_orientation": [4],
        "physics_gt.object_velocity": [3],
        "physics_gt.object_angular_velocity": [3],
        "physics_gt.object_acceleration": [3],
        "physics_gt.ee_to_object_distance": [1],
        "physics_gt.contact_flag": [1],
        "physics_gt.contact_force": [3],
        "physics_gt.object_on_surface": [1],
        "physics_gt.contact_point": [3],
        "physics_gt.phase": [1],
    }
    if task_name in ("pick_place", "push"):
        physics_gt_shapes["physics_gt.target_position"] = [3]
    if task_name == "stack":
        physics_gt_shapes.update({
            "physics_gt.object_1_position": [3],
            "physics_gt.object_1_orientation": [4],
            "physics_gt.object_1_velocity": [3],
            "physics_gt.object_1_angular_velocity": [3],
            "physics_gt.object_1_acceleration": [3],
            "physics_gt.ee_to_object_1_distance": [1],
            "physics_gt.object_0_to_object_1_distance": [1],
            "physics_gt.object_1_on_surface": [1],
            "physics_gt.object_object_contact_flag": [1],
            "physics_gt.object_object_contact_force": [3],
            "physics_gt.object_object_contact_point": [3],
        })

    info = {
        "codebase_version": "v2.0",
        "robot_type": "franka",
        "total_episodes": num_episodes,
        "total_frames": total_frames,
        "total_tasks": len(tasks_meta_list),
        "total_videos": num_episodes * 2,  # 2 cameras
        "total_chunks": (num_episodes - 1) // CHUNKS_SIZE + 1,
        "chunks_size": CHUNKS_SIZE,
        "fps": fps,
        "splits": {"train": f"0:{num_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.images.image_0": {
                "dtype": "video",
                "shape": [384, 384, 3],
                "names": ["height", "width", "rgb"],
                "info": {
                    "video.fps": float(fps),
                    "video.height": 384,
                    "video.width": 384,
                    "video.channels": 3,
                    "video.codec": VIDEO_CODEC.replace("lib", ""),
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            },
            "observation.images.image_1": {
                "dtype": "video",
                "shape": [384, 384, 3],
                "names": ["height", "width", "rgb"],
                "info": {
                    "video.fps": float(fps),
                    "video.height": 384,
                    "video.width": 384,
                    "video.channels": 3,
                    "video.codec": VIDEO_CODEC.replace("lib", ""),
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [8],
                "names": {"motors": ["x", "y", "z", "roll", "pitch", "yaw", "pad", "gripper"]},
            },
            "action": {
                "dtype": "float32",
                "shape": [7],
                "names": {"motors": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]},
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
            "next.done": {"dtype": "bool", "shape": [1], "names": None},
            "next.reward": {"dtype": "float32", "shape": [1], "names": None},
        },
    }

    # Add physics GT features to info
    for k, shape in physics_gt_shapes.items():
        info["features"][k] = {"dtype": "float32", "shape": shape, "names": None}

    with open(os.path.join(meta_dir, "info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"\nSaved meta/info.json")

    # episodes.jsonl
    with open(os.path.join(meta_dir, "episodes.jsonl"), "w") as f:
        for ep in episodes_meta:
            f.write(json.dumps(ep) + "\n")
    print(f"Saved meta/episodes.jsonl ({len(episodes_meta)} episodes)")

    # tasks.jsonl
    with open(os.path.join(meta_dir, "tasks.jsonl"), "w") as f:
        for t in tasks_meta_list:
            f.write(json.dumps(t) + "\n")
    print(f"Saved meta/tasks.jsonl ({len(tasks_meta_list)} tasks)")

    # stats.json
    states_arr = np.array(all_episode_stats["observation.state"])
    actions_arr = np.array(all_episode_stats["action"])
    timestamps_arr = np.array(all_episode_stats["timestamp"])

    def compute_stats(arr):
        return {
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).tolist(),
            "min": arr.min(axis=0).tolist(),
            "max": arr.max(axis=0).tolist(),
            "q01": np.percentile(arr, 1, axis=0).tolist(),
            "q99": np.percentile(arr, 99, axis=0).tolist(),
        }

    stats = {
        "observation.state": compute_stats(states_arr),
        "action": compute_stats(actions_arr),
        "timestamp": compute_stats(timestamps_arr.reshape(-1, 1)),
    }

    # Add physics GT stats
    for gt_key in physics_gt_keys:
        all_gt_vals = []
        for ep_data in all_episodes_physics_gt:
            all_gt_vals.extend(ep_data[gt_key])
        if all_gt_vals:
            gt_arr = np.array(all_gt_vals)
            if gt_arr.ndim == 1:
                gt_arr = gt_arr.reshape(-1, 1)
            stats[gt_key] = compute_stats(gt_arr)

    with open(os.path.join(meta_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved meta/stats.json")

    # modality.json (matching BridgeData structure)
    modality = {
        "state": {
            "x": {"start": 0, "end": 1}, "y": {"start": 1, "end": 2},
            "z": {"start": 2, "end": 3}, "roll": {"start": 3, "end": 4},
            "pitch": {"start": 4, "end": 5}, "yaw": {"start": 5, "end": 6},
            "pad": {"start": 6, "end": 7}, "gripper": {"start": 7, "end": 8},
        },
        "action": {
            "x": {"start": 0, "end": 1}, "y": {"start": 1, "end": 2},
            "z": {"start": 2, "end": 3}, "roll": {"start": 3, "end": 4},
            "pitch": {"start": 4, "end": 5}, "yaw": {"start": 5, "end": 6},
            "gripper": {"start": 6, "end": 7},
        },
        "video": {
            "image_0": {"original_key": "observation.images.image_0"},
            "image_1": {"original_key": "observation.images.image_1"},
        },
        "annotation": {
            "human.action.task_description": {"original_key": "task_index"},
        },
    }
    with open(os.path.join(meta_dir, "modality.json"), "w") as f:
        json.dump(modality, f, indent=2)
    print(f"Saved meta/modality.json")

    print(f"\n{'='*60}")
    print(f"Data collection complete for task: {task_name}")
    print(f"  Episodes: {num_episodes}")
    print(f"  Total frames: {total_frames}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")

    env.close()


def main():
    output_dir = os.path.join(args_cli.output_dir, args_cli.task)
    collect_task(args_cli.task, args_cli.num_episodes, output_dir, use_oracle=args_cli.use_oracle)
    simulation_app.close()


if __name__ == "__main__":
    main()
