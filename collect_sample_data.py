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
parser.add_argument("--task", type=str, required=True, choices=["lift", "pick_place", "push", "stack", "strike", "drawer", "reach", "peg_insert", "nut_thread"])
parser.add_argument("--num_episodes", type=int, default=5)
parser.add_argument("--use_oracle", action="store_true", help="Use scripted oracle policy instead of random actions")
parser.add_argument("--step0", action="store_true", help="Use Step 0 scripted policy (no target, random direction)")
parser.add_argument("--filter_success", action="store_true", help="Only save successful episodes (retry on failure)")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel envs (>1 uses parallel collection)")
parser.add_argument("--rl_checkpoint", type=str, default=None, help="Path to RL checkpoint (.pt for RSL-RL, .pth for RL-Games)")
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
from physrepa_tasks.envs.strike_env_cfg import PhysREPAStrikeEnvCfg
from physrepa_tasks.envs.drawer_env_cfg import PhysREPADrawerEnvCfg
from physrepa_tasks.envs.reach_env_cfg import PhysREPAReachEnvCfg
from physrepa_tasks.envs.peg_insert_env_cfg import PhysREPAPegInsertEnvCfg
from physrepa_tasks.envs.nut_thread_env_cfg import PhysREPANutThreadEnvCfg
from physrepa_tasks.policies.scripted_policy import (
    LiftPolicy, PickPlacePolicy, PushPolicy, StackPolicy,
    ReachPolicy, StrikePolicy, DrawerPolicy, PegInsertPolicy, NutThreadPolicy,
    Step0PushPolicy, Step0StrikePolicy,
)

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
    "push": "Push the red cube to the blue zone",
    "stack": "Stack the small red cube on top of the large blue cube",
    "strike": "Hit the red ball to the blue zone",
    "drawer": "Open the drawer",
    "reach": "Reach the target position",
    "peg_insert": "Insert the peg into the hole",
    "nut_thread": "Thread the nut onto the bolt",
}

TASK_CONFIGS = {
    "lift": PhysREPALiftEnvCfg,
    "pick_place": PhysREPAPickPlaceEnvCfg,
    "push": PhysREPAPushEnvCfg,
    "stack": PhysREPAStackEnvCfg,
    "strike": PhysREPAStrikeEnvCfg,
    "drawer": PhysREPADrawerEnvCfg,
    "reach": PhysREPAReachEnvCfg,
    "peg_insert": PhysREPAPegInsertEnvCfg,
    "nut_thread": PhysREPANutThreadEnvCfg,
}

TASK_POLICIES = {
    "lift": LiftPolicy,
    "pick_place": PickPlacePolicy,
    "push": PushPolicy,
    "stack": StackPolicy,
    "reach": ReachPolicy,
    "strike": StrikePolicy,
    "drawer": DrawerPolicy,
    "peg_insert": PegInsertPolicy,
    "nut_thread": NutThreadPolicy,
}

# Map policy state index → phase label index for each task
TASK_STATE_TO_PHASE = {
    "lift": {0: 0, 1: 0, 2: 1, 3: 2, 4: 7},  # approach→reach, descend→reach, grasp, lift, hold→idle
    "pick_place": {0: 0, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6},  # reach, reach, grasp, lift, transport, place, release, retract
    "push": {0: 0, 1: 0, 2: 5, 3: 7},  # approach→reach, descend→reach, push(=5), done→idle
    "stack": {0: 0, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6},  # reach, reach, grasp, lift, transport, place, release, retract
    "reach": {0: 0, 1: 7},  # move→reach, hold→idle
    "strike": {0: 0, 1: 0, 2: 5, 3: 6, 4: 7},  # position→reach, descend→reach, swing(=5 release), retract, wait→idle
    "drawer": {0: 0, 1: 0, 2: 1, 3: 5, 4: 6},  # approach→reach, reach_handle→reach, grasp, pull(=5 release), done→retract
    "peg_insert": {0: 0, 1: 0, 2: 4, 3: 7},  # approach→reach, align→reach, insert(=4 place), hold→idle
    "nut_thread": {0: 0, 1: 0, 2: 4, 3: 5, 4: 7},  # approach→reach, descend→reach, engage(=4), rotate(=5), hold→idle
}

# Object size constants for on_surface checks
TABLE_SURFACE_Z = 0.0
CUBE_HALF_SIZE = 0.03  # 0.06m cube / 2 (lift, pick_place, push)
BALL_RADIUS = 0.03  # strike ball radius
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
    elif task_name == "drawer":
        params["task_type"] = "drawer"
        try:
            cabinet = env.scene["cabinet"]
            joint_idx = cabinet.find_joints("drawer_top_joint")[0]
            # Joint damping
            dampings = cabinet.root_physx_view.get_dof_dampings()
            params["drawer_joint_damping"] = round(dampings[0, joint_idx[0]].item(), 4)
            # Handle body mass
            handle_body_idx = cabinet.body_names.index("drawer_handle_top")
            masses = cabinet.root_physx_view.get_masses()
            params["drawer_handle_mass"] = round(masses[0, handle_body_idx].item(), 4)
            # Handle friction
            mat = cabinet.root_physx_view.get_material_properties()
            params["handle_static_friction"] = round(mat[0, handle_body_idx, 0].item(), 4)
            params["handle_dynamic_friction"] = round(mat[0, handle_body_idx, 1].item(), 4)
        except Exception as e:
            print(f"  [WARN] failed to read drawer physics params: {e}")
    elif task_name == "reach":
        # Reach task: no objects
        params["task_type"] = "reach"
    elif task_name == "peg_insert":
        # Factory env: held_asset = peg, fixed_asset = hole
        held = env._held_asset if hasattr(env, '_held_asset') else env.scene["peg"]
        fixed = env._fixed_asset if hasattr(env, '_fixed_asset') else env.scene["hole"]
        mat_held = held.root_physx_view.get_material_properties()
        mat_fixed = fixed.root_physx_view.get_material_properties()
        params["peg_static_friction"] = round(mat_held[0, 0, 0].item(), 4)
        params["peg_dynamic_friction"] = round(mat_held[0, 0, 1].item(), 4)
        params["peg_mass"] = round(held.root_physx_view.get_masses()[0, 0].item(), 6)
        params["hole_static_friction"] = round(mat_fixed[0, 0, 0].item(), 4)
        params["hole_dynamic_friction"] = round(mat_fixed[0, 0, 1].item(), 4)
        params["task_type"] = "peg_insert"
    elif task_name == "nut_thread":
        held = env._held_asset if hasattr(env, '_held_asset') else env.scene["nut"]
        fixed = env._fixed_asset if hasattr(env, '_fixed_asset') else env.scene["bolt"]
        mat_held = held.root_physx_view.get_material_properties()
        mat_fixed = fixed.root_physx_view.get_material_properties()
        params["nut_static_friction"] = round(mat_held[0, 0, 0].item(), 4)
        params["nut_dynamic_friction"] = round(mat_held[0, 0, 1].item(), 4)
        params["nut_mass"] = round(held.root_physx_view.get_masses()[0, 0].item(), 6)
        params["bolt_static_friction"] = round(mat_fixed[0, 0, 0].item(), 4)
        params["bolt_dynamic_friction"] = round(mat_fixed[0, 0, 1].item(), 4)
        params["task_type"] = "nut_thread"
    else:
        obj = env.scene["object"]
        mass = obj.root_physx_view.get_masses()[0, 0].item()
        mat_props = obj.root_physx_view.get_material_properties()
        params["object_0_mass"] = round(mass, 4)
        params["object_0_static_friction"] = round(mat_props[0, 0, 0].item(), 4)
        params["object_0_dynamic_friction"] = round(mat_props[0, 0, 1].item(), 4)
        params["object_0_color"] = "red"
        params["object_0_type"] = "ball" if task_name == "strike" else "cube"

    if task_name in ("push", "strike"):
        surface = env.scene["surface"]
        surface_mat = surface.root_physx_view.get_material_properties()
        params["surface_static_friction"] = round(surface_mat[0, 0, 0].item(), 4)
        params["surface_dynamic_friction"] = round(surface_mat[0, 0, 1].item(), 4)
    if task_name == "strike":
        # Also record restitution for strike (ball bounce)
        mat_props = env.scene["object"].root_physx_view.get_material_properties()
        params["object_0_restitution"] = round(mat_props[0, 0, 2].item(), 4)

    return params


def collect_task(task_name: str, num_episodes: int, output_dir: str, use_oracle: bool = False,
                  rl_checkpoint: str | None = None, step0: bool = False, filter_success: bool = False):
    """Collect episodes for a single task and save in LeRobot V2 format."""
    is_factory = task_name in ("peg_insert", "nut_thread")
    is_rl_wrapped = False
    rl_policy = None
    _drawer_env_wrapped = None

    if is_factory:
        from physrepa_tasks.envs.factory_camera_env import (
            FactoryCameraEnv, PegInsertCameraCfg, NutThreadCameraCfg,
        )
        cfg = PegInsertCameraCfg() if task_name == "peg_insert" else NutThreadCameraCfg()
        cfg.scene.num_envs = 1
        env = FactoryCameraEnv(cfg=cfg)

        # Load RL-Games checkpoint for Factory tasks
        if rl_checkpoint is not None:
            from physrepa_tasks.utils.rl_games_policy import RlGamesPolicy
            rl_policy = RlGamesPolicy(rl_checkpoint, device=env.device)
            print(f"  Loaded RL-Games checkpoint: {rl_checkpoint}")
    elif rl_checkpoint is not None and task_name == "drawer":
        # Drawer: use Isaac Lab official RL env with cameras added
        import gymnasium as gym
        import isaaclab.sim as sim_utils
        import isaaclab_tasks  # register official envs
        import importlib
        gym_id = "Isaac-Open-Drawer-Franka-v0"
        spec = gym.spec(gym_id)
        env_cfg_entry = spec.kwargs["env_cfg_entry_point"]
        if isinstance(env_cfg_entry, str):
            mod_path, cls_name = env_cfg_entry.rsplit(":", 1)
            mod = importlib.import_module(mod_path)
            cfg = getattr(mod, cls_name)()
        else:
            cfg = env_cfg_entry()
        cfg.scene.num_envs = 1
        cfg.observations.policy.enable_corruption = False
        # Disable debug visualization arrows
        if hasattr(cfg.scene, 'cabinet_frame'):
            cfg.scene.cabinet_frame.debug_vis = False
        # Add cameras to scene
        from isaaclab.sensors import CameraCfg
        cfg.scene.table_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/table_cam", update_period=0.0, height=384, width=384,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=18.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 5.0)),
            offset=CameraCfg.OffsetCfg(
                pos=(-0.2, 0.9, 0.8), rot=(0.16560, -0.23780, 0.78541, -0.54695), convention="ros"),
        )
        cfg.scene.wrist_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam", update_period=0.0, height=384, width=384,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 2.0)),
            offset=CameraCfg.OffsetCfg(
                pos=(0.13, 0.0, -0.15), rot=(-0.70614, 0.03701, 0.03701, -0.70614), convention="ros"),
        )
        env = gym.make(gym_id, cfg=cfg)
        # Load RSL-RL policy
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner
        agent_cfg_entry = spec.kwargs["rsl_rl_cfg_entry_point"]
        mod_path, cls_name = agent_cfg_entry.rsplit(":", 1)
        mod = importlib.import_module(mod_path)
        agent_cfg = getattr(mod, cls_name)()
        _drawer_env_wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        ppo_runner = OnPolicyRunner(_drawer_env_wrapped, agent_cfg.to_dict(), log_dir=None, device=env.unwrapped.device)
        ppo_runner.load(rl_checkpoint)
        rl_policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
        # Use unwrapped for scene access, wrapped for stepping
        env = env.unwrapped
        is_rl_wrapped = True
        print(f"  Loaded RSL-RL Drawer checkpoint: {rl_checkpoint}")
    elif rl_checkpoint is not None and task_name in ("push", "strike"):
        # Push/Strike: use RL env (IK relative) with cameras added
        import gymnasium as gym
        import isaaclab.sim as sim_utils
        import importlib
        import physrepa_tasks.rl_envs  # register envs
        gym_id = {"push": "PhysREPA-Push-Franka-v0", "strike": "PhysREPA-Strike-Franka-v0"}[task_name]
        spec = gym.spec(gym_id)
        env_cfg_entry = spec.kwargs["env_cfg_entry_point"]
        if isinstance(env_cfg_entry, str):
            mod_path, cls_name = env_cfg_entry.rsplit(":", 1)
            mod = importlib.import_module(mod_path)
            cfg = getattr(mod, cls_name)()
        else:
            cfg = env_cfg_entry()
        cfg.scene.num_envs = 1
        cfg.observations.policy.enable_corruption = False
        # Add cameras + contact sensors for physics GT
        from isaaclab.sensors import CameraCfg, ContactSensorCfg
        cfg.scene.table_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/table_cam", update_period=0.0, height=384, width=384,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 4.0)),
            offset=CameraCfg.OffsetCfg(
                pos=(1.6, 0.0, 0.90), rot=(0.33900, -0.62054, -0.62054, 0.33900), convention="ros"),
        )
        cfg.scene.wrist_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam", update_period=0.0, height=384, width=384,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 2.0)),
            offset=CameraCfg.OffsetCfg(
                pos=(0.13, 0.0, -0.15), rot=(-0.70614, 0.03701, 0.03701, -0.70614), convention="ros"),
        )
        # Contact sensor: finger ↔ object
        cfg.scene.contact_sensor = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
            update_period=0.0, history_length=1,
            track_air_time=False, track_contact_points=True,
            force_threshold=0.5,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        )
        # Contact sensor: object ↔ surface (push/strike have Surface)
        if hasattr(cfg.scene, "surface"):
            cfg.scene.object_surface_contact = ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Object",
                update_period=0.0, history_length=1,
                track_air_time=False, track_contact_points=False,
                force_threshold=0.5,
                filter_prim_paths_expr=["{ENV_REGEX_NS}/Surface"],
            )
        env = gym.make(gym_id, cfg=cfg)
        # Load RSL-RL policy
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner
        agent_cfg_entry = spec.kwargs["rsl_rl_cfg_entry_point"]
        mod_path, cls_name = agent_cfg_entry.rsplit(":", 1)
        mod = importlib.import_module(mod_path)
        agent_cfg = getattr(mod, cls_name)()
        _drawer_env_wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        ppo_runner = OnPolicyRunner(_drawer_env_wrapped, agent_cfg.to_dict(), log_dir=None, device=env.unwrapped.device)
        ppo_runner.load(rl_checkpoint)
        rl_policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
        env = env.unwrapped
        is_rl_wrapped = True  # reuse same stepping logic as drawer
        print(f"  Loaded RSL-RL {task_name} checkpoint: {rl_checkpoint}")
    else:
        cfg_cls = TASK_CONFIGS[task_name]
        cfg = cfg_cls()
        cfg.scene.num_envs = 1
        # Step0: remove target marker (no target needed)
        if step0 and hasattr(cfg.scene, 'target_marker'):
            cfg.scene.target_marker = None
        env = ManagerBasedRLEnv(cfg=cfg)

    print(f"\n{'='*60}")
    print(f"Collecting {num_episodes} episodes for task: {task_name}")
    if is_factory:
        print(f"  (Factory Direct env with cameras)")
    print(f"{'='*60}")

    robot = env.scene["robot"] if not is_factory else env._robot
    try:
        finger_ids = robot.find_joints("panda_finger.*")[0] if not is_factory else None
    except Exception:
        finger_ids = None  # UR10 has no fingers

    # Get EE frame sensor (not available in Factory env)
    ee_frame = env.scene["ee_frame"] if not is_factory else None

    # Derive FPS and dt from config
    dt = cfg.sim.dt * cfg.decimation
    fps = int(1.0 / dt)

    # Oracle policy
    oracle_policy = None
    if step0 and task_name in ("push", "strike"):
        step0_policies = {"push": Step0PushPolicy, "strike": Step0StrikePolicy}
        policy_cls = step0_policies[task_name]
        oracle_policy = policy_cls(num_envs=1, device=env.device)
        print(f"  Using Step 0 policy: {policy_cls.__name__} (no target, random direction)")
    elif use_oracle:
        policy_cls = TASK_POLICIES[task_name]
        oracle_policy = policy_cls(num_envs=1, device=env.device)
        print(f"  Using oracle policy: {policy_cls.__name__}")

    episodes_meta = []
    tasks_meta_dict = {}  # task_description -> task_index
    tasks_meta_list = []
    global_index = 0
    all_episode_stats = {"observation.state": [], "action": [], "timestamp": []}

    # Physics GT column names — task-specific
    # Common EE columns for all tasks
    _common_ee_keys = [
        "physics_gt.ee_position",
        "physics_gt.ee_orientation",
        "physics_gt.ee_velocity",
        "physics_gt.ee_angular_velocity",
        "physics_gt.ee_acceleration",
    ]

    if task_name == "reach":
        physics_gt_keys = _common_ee_keys + [
            "physics_gt.target_position",
            "physics_gt.ee_to_target_distance",
            "physics_gt.phase",
        ]
    elif task_name == "peg_insert":
        physics_gt_keys = _common_ee_keys + [
            "physics_gt.peg_position",
            "physics_gt.peg_orientation",
            "physics_gt.peg_velocity",
            "physics_gt.peg_angular_velocity",
            "physics_gt.hole_position",
            "physics_gt.contact_flag",
            "physics_gt.contact_force",
            "physics_gt.contact_point",
            # Pair-specific contacts
            "physics_gt.contact_finger_l_peg_flag",
            "physics_gt.contact_finger_l_peg_force",
            "physics_gt.contact_finger_r_peg_flag",
            "physics_gt.contact_finger_r_peg_force",
            "physics_gt.contact_peg_socket_flag",
            "physics_gt.contact_peg_socket_force",
            # Task-specific raw
            "physics_gt.insertion_depth",
            "physics_gt.peg_hole_lateral_error",
            "physics_gt.phase",
        ]
    elif task_name == "nut_thread":
        physics_gt_keys = _common_ee_keys + [
            "physics_gt.nut_position",
            "physics_gt.nut_orientation",
            "physics_gt.nut_velocity",
            "physics_gt.nut_angular_velocity",
            "physics_gt.bolt_position",
            "physics_gt.bolt_orientation",
            "physics_gt.contact_flag",
            "physics_gt.contact_force",
            "physics_gt.contact_point",
            # Pair-specific contacts
            "physics_gt.contact_finger_l_nut_flag",
            "physics_gt.contact_finger_l_nut_force",
            "physics_gt.contact_finger_r_nut_flag",
            "physics_gt.contact_finger_r_nut_force",
            "physics_gt.contact_nut_bolt_flag",
            "physics_gt.contact_nut_bolt_force",
            # Task-specific raw
            "physics_gt.axial_progress",
            "physics_gt.nut_bolt_relative_angle",
            "physics_gt.phase",
        ]
    elif task_name == "drawer":
        physics_gt_keys = _common_ee_keys + [
            "physics_gt.drawer_joint_pos",
            "physics_gt.drawer_joint_vel",
            "physics_gt.handle_position",
            "physics_gt.handle_velocity",
            "physics_gt.contact_flag",
            "physics_gt.contact_force",
            "physics_gt.contact_point",
            # Pair-specific: finger L/R ↔ handle
            "physics_gt.contact_finger_l_handle_flag",
            "physics_gt.contact_finger_l_handle_force",
            "physics_gt.contact_finger_r_handle_flag",
            "physics_gt.contact_finger_r_handle_force",
            # Task-specific raw
            "physics_gt.drawer_opening_extent",
            "physics_gt.phase",
        ]
    else:
        # Push, Strike, Lift, PickPlace, Stack
        physics_gt_keys = _common_ee_keys + [
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
            # Pair-specific: EE ↔ object (uses existing contact_sensor)
            "physics_gt.contact_finger_l_object_flag",
            "physics_gt.contact_finger_l_object_force",
            "physics_gt.phase",
        ]

        if task_name in ("push", "strike"):
            physics_gt_keys += [
                "physics_gt.target_position",
                # Pair-specific: object ↔ surface
                "physics_gt.contact_object_surface_flag",
                "physics_gt.contact_object_surface_force",
                # Task-specific raw
                "physics_gt.object_to_target_distance",
            ]
        elif task_name in ("pick_place",):
            physics_gt_keys.append("physics_gt.target_position")

        if task_name == "strike":
            physics_gt_keys.append("physics_gt.ball_planar_travel_distance")

        # Additional keys for stack task
        if task_name == "stack":
            physics_gt_keys += [
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

    all_episodes_physics_gt = []

    saved_episodes = 0
    attempted_episodes = 0
    max_attempts = num_episodes * 5 if filter_success else num_episodes  # cap retries

    while saved_episodes < num_episodes and attempted_episodes < max_attempts:
        ep_idx = saved_episodes
        attempted_episodes += 1
        print(f"\n--- Episode {ep_idx} (attempt {attempted_episodes}) ---")

        # Reset with overlap check for pick_place/push
        # Ensure object and target are at least 0.1m apart
        for _reset_attempt in range(10):
            if is_rl_wrapped:
                obs_w, info = _drawer_env_wrapped.get_observations()
                obs = {"policy": obs_w}
                # Trigger reset by getting observations after env auto-resets
                env.reset()
            else:
                obs, info = env.reset()
            if task_name in ("pick_place", "push", "strike") and not is_factory:
                # Read object/target positions (from obs or env.scene for RL)
                if "physics_gt" in obs:
                    gt = obs["physics_gt"]
                    obj_pos = gt["object_position"][0, :2].cpu()
                    target_pos = gt["target_position"][0, :2].cpu()
                else:
                    # RL env: read from scene directly
                    _obj = env.scene["object"]
                    obj_pos = (_obj.data.root_pos_w[0, :2] - env.scene.env_origins[0, :2]).cpu()
                    try:
                        _cmd = env.command_manager.get_command("object_pose")[0, :2]
                        target_pos = _cmd.cpu()
                    except Exception:
                        break  # can't check overlap, just proceed
                dist = torch.norm(obj_pos - target_pos).item()
                if dist >= 0.15:
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

        # Randomize physics for Factory tasks (friction, mass) — PEZ probing
        if is_factory and hasattr(env, 'randomize_physics'):
            factory_phys = env.randomize_physics()
            print(f"  Factory physics randomized: {factory_phys}")

        # Read physics params after reset (when randomization has been applied)
        physics_params = read_physics_params(env, task_name)
        print(f"  Physics params: {physics_params}")

        # Set adaptive parameters based on friction
        if task_name in ("push", "strike") and oracle_policy is not None and hasattr(oracle_policy, 'set_friction'):
            sf = physics_params.get('surface_static_friction', 0.5)
            of = physics_params.get('object_0_static_friction', 0.5)
            oracle_policy.set_friction(sf, of)
            print(f"  {task_name} adaptive: surf_f={sf:.3f}, obj_f={of:.3f}")

        # Storage for this episode
        ep_states = []
        ep_actions = []
        ep_timestamps = []
        ep_physics_gt = {k: [] for k in physics_gt_keys}
        ep_frames_table = []
        ep_frames_wrist = []
        ep_rewards = []

        # Sync target marker to command target position
        if not is_factory:
            if task_name in ("pick_place", "push", "strike"):
                from physrepa_tasks.mdp.sync_marker import sync_target_marker
                sync_target_marker(env, env_ids=None, command_name="object_pose")
            elif task_name == "reach":
                from physrepa_tasks.mdp.sync_marker import sync_target_marker
                sync_target_marker(env, env_ids=None, command_name="ee_pose", fixed_z=None)

        # Warmup: 0.5 seconds of no-action steps for rendering stabilization
        if task_name == "reach" or is_factory:
            action_dim = 6
        elif rl_checkpoint is not None and task_name == "drawer":
            action_dim = 8
        elif step0 and task_name == "push":
            action_dim = 4  # position-only IK (3D) + gripper (1D)
        elif step0 and task_name == "strike":
            action_dim = 4  # position-only IK (3D) + gripper (1D)
        else:
            action_dim = 7
        if is_factory:
            # Factory: render-only warmup (no physics step, no episode budget consumed)
            for _ in range(3):
                env.sim.render()
        else:
            warmup_steps = int(0.5 / dt)
            for _ in range(warmup_steps):
                if is_rl_wrapped:
                    obs_w, _, _, _ = _drawer_env_wrapped.step(torch.zeros(1, action_dim, device=env.device))
                    obs = {"policy": obs_w}
                else:
                    obs, _, _, _, _ = env.step(torch.zeros(1, action_dim, device=env.device))

        # Full episode length (warmup is separate, doesn't subtract from data)
        episode_s = 5.0 if step0 else cfg.episode_length_s
        max_steps = int(episode_s / dt)

        # Reset oracle/RL policy after warmup
        if oracle_policy is not None:
            oracle_policy.reset()
        if rl_policy is not None and hasattr(rl_policy, 'reset'):
            rl_policy.reset()

        # Previous velocities for acceleration computation
        prev_ee_vel = None
        prev_obj_vel = None
        prev_obj1_vel = None  # for stack task cube_b
        prev_ball_pos = None  # for strike ball_planar_travel_distance
        ball_rolling_dist = 0.0

        for step in range(max_steps):
            # Generate action
            if rl_policy is not None:
                with torch.inference_mode():
                    if is_factory:
                        # rl_games LSTM: takes flat obs from Factory env
                        flat_obs = obs["policy"] if isinstance(obs, dict) else obs
                        action = rl_policy(flat_obs)
                    else:
                        # RSL-RL: takes flat concatenated obs
                        flat_obs = obs["policy"] if isinstance(obs, dict) else obs
                        if isinstance(flat_obs, dict):
                            # Concatenate if dict (shouldn't happen for RL envs)
                            flat_obs = torch.cat([v.flatten() for v in flat_obs.values()]).unsqueeze(0)
                        action = rl_policy(flat_obs)
                    if action.dim() == 1:
                        action = action.unsqueeze(0)
            elif oracle_policy is not None:
                action = oracle_policy.get_action(obs)
            else:
                action = torch.zeros(1, action_dim, device=env.device)
                if is_factory:
                    action[0, :6] = torch.randn(6, device=env.device) * 0.5
                else:
                    action[0, :6] = torch.randn(6, device=env.device) * 0.1
                    if action_dim == 7:
                        action[0, 6] = 1.0 if step < max_steps // 2 else -1.0

            # Build physics GT dict (BEFORE step — reads current state)
            if is_factory:
                phys = env.get_physics_data()
                # Build a fake gt dict for uniform access below
                gt = {
                    "ee_position": phys["ee_position"],
                    "ee_orientation": phys["ee_orientation"],
                    "ee_velocity": phys["ee_velocity"],
                    "ee_angular_velocity": phys["ee_angular_velocity"],
                    "peg_position": phys["held_position"],
                    "peg_orientation": phys["held_orientation"],
                    "peg_velocity": phys["held_velocity"],
                    "peg_angular_velocity": phys["held_angular_velocity"],
                    "hole_position": phys["fixed_position"],
                    "nut_position": phys["held_position"],
                    "nut_orientation": phys["held_orientation"],
                    "nut_velocity": phys["held_velocity"],
                    "nut_angular_velocity": phys["held_angular_velocity"],
                    "bolt_position": phys["fixed_position"],
                    "bolt_orientation": phys["fixed_orientation"],
                    # Pair-specific contacts
                    "finger_l_contact_flag": phys["finger_l_contact_flag"],
                    "finger_l_contact_force": phys["finger_l_contact_force"],
                    "finger_r_contact_flag": phys["finger_r_contact_flag"],
                    "finger_r_contact_force": phys["finger_r_contact_force"],
                    "held_fixed_contact_flag": phys["held_fixed_contact_flag"],
                    "held_fixed_contact_force": phys["held_fixed_contact_force"],
                    "contact_flag": phys["contact_flag"],
                    "contact_force": phys["contact_force"],
                }
            elif "physics_gt" in obs:
                gt = obs["physics_gt"]
            else:
                # RL env without physics_gt (e.g., drawer RL) — read from env directly
                _robot = env.scene["robot"]
                _ee_frame = env.scene["ee_frame"]
                _hand_idx = _robot.body_names.index("panda_hand")
                _ee_pos = _ee_frame.data.target_pos_w[0, 0, :] - env.scene.env_origins[0]
                _ee_quat = _robot.data.body_quat_w[0, _hand_idx]
                _ee_vel = _robot.data.body_lin_vel_w[0, _hand_idx]
                _ee_angvel = _robot.data.body_ang_vel_w[0, _hand_idx]
                # Read contact from contact_sensor if available, otherwise zeros
                if "contact_sensor" in env.scene.sensors:
                    _csensor = env.scene.sensors["contact_sensor"]
                    _cflag_t = _csensor.data.net_forces_w[0, 0, :]
                    _cflag_val = (torch.norm(_cflag_t) > 0.5).float()
                    _cforce = _cflag_t
                else:
                    _cflag_val = torch.tensor(0.0, device=env.device)
                    _cforce = torch.zeros(3, device=env.device)
                gt = {
                    "ee_position": _ee_pos.unsqueeze(0),
                    "ee_orientation": _ee_quat.unsqueeze(0),
                    "ee_velocity": _ee_vel.unsqueeze(0),
                    "ee_angular_velocity": _ee_angvel.unsqueeze(0),
                    "contact_flag": _cflag_val.unsqueeze(0).unsqueeze(0),
                    "contact_force": _cforce.unsqueeze(0),
                }
                if task_name in ("push", "strike"):
                    # Push/Strike RL fallback: read object state from env.scene
                    _obj = env.scene["object"]
                    _obj_pos = _obj.data.root_pos_w[0] - env.scene.env_origins[0]
                    _obj_quat = _obj.data.root_quat_w[0]
                    _obj_vel = _obj.data.root_lin_vel_w[0]
                    _obj_angvel = _obj.data.root_ang_vel_w[0]
                    gt["object_position"] = _obj_pos.unsqueeze(0)
                    gt["object_orientation"] = _obj_quat.unsqueeze(0)
                    gt["object_velocity"] = _obj_vel.unsqueeze(0)
                    gt["object_angular_velocity"] = _obj_angvel.unsqueeze(0)
                    gt["ee_to_object_distance"] = torch.norm(_ee_pos - _obj_pos).unsqueeze(0).unsqueeze(0)
                    # Target position from command manager (base frame → world frame)
                    try:
                        _cmd = env.command_manager.get_command("object_pose")[0, :3]
                        _root_pos = _robot.data.root_pos_w[0]
                        _root_quat = _robot.data.root_quat_w[0]
                        from isaaclab.utils.math import combine_frame_transforms
                        _target_w, _ = combine_frame_transforms(
                            _root_pos.unsqueeze(0), _root_quat.unsqueeze(0), _cmd.unsqueeze(0)
                        )
                        _target_local = _target_w[0] - env.scene.env_origins[0]
                        gt["target_position"] = _target_local.unsqueeze(0)
                    except Exception:
                        gt["target_position"] = torch.zeros(1, 3, device=env.device)
                elif task_name == "drawer":
                    _cabinet = env.scene["cabinet"]
                    _jpos = _cabinet.data.joint_pos[0, _cabinet.find_joints("drawer_top_joint")[0]].unsqueeze(0)
                    _jvel = _cabinet.data.joint_vel[0, _cabinet.find_joints("drawer_top_joint")[0]].unsqueeze(0)
                    gt["drawer_joint_pos"] = _jpos.unsqueeze(0)
                    gt["drawer_joint_vel"] = _jvel.unsqueeze(0)
                    # Handle world position from cabinet_frame sensor
                    if "cabinet_frame" in env.scene.sensors:
                        _cab_frame = env.scene.sensors["cabinet_frame"]
                        _handle_pos = _cab_frame.data.target_pos_w[0, 0, :] - env.scene.env_origins[0]
                        gt["handle_position"] = _handle_pos.unsqueeze(0)
                        _handle_body_idx = _cabinet.body_names.index("drawer_handle_top")
                        _handle_vel = _cabinet.data.body_lin_vel_w[0, _handle_body_idx]
                        gt["handle_velocity"] = _handle_vel.unsqueeze(0)

            # EE state: convert to BridgeData 8D format [x,y,z,roll,pitch,yaw,pad,gripper]
            if is_factory:
                ee_pos_b_np = phys["ee_position"][0].cpu().numpy()
                # Factory fingertip quat
                hand_idx = robot.body_names.index("panda_hand")
                ee_quat_b_np = robot.data.body_quat_w[0, hand_idx].cpu().numpy()
                gripper_val = robot.data.joint_pos[0, -1].item() / 0.04
            else:
                ee_pos_b_np, ee_quat_b_np = get_ee_pose_in_base_frame(env)
                gripper_val = robot.data.joint_pos[0, finger_ids].mean().item() / 0.04 if finger_ids is not None else 0.0
            state_8d = ee_state_to_8d(ee_pos_b_np, ee_quat_b_np, gripper_val)
            ep_states.append(state_8d)

            # Action: always store as 7D [dx, dy, dz, droll, dpitch, dyaw, gripper]
            raw_action = action[0].cpu().numpy().astype(np.float32)
            if action_dim == 3:
                # UR10 position-only: [dx,dy,dz] → pad to 7D
                action_7d = np.zeros(7, dtype=np.float32)
                action_7d[:3] = raw_action[:3]
            elif action_dim == 4:
                # Position-only IK: [dx,dy,dz, grip] → pad to 7D
                action_7d = np.zeros(7, dtype=np.float32)
                action_7d[:3] = raw_action[:3]
                action_7d[6] = 0.0 if raw_action[3] < 0 else 1.0
            elif action_dim == 6:
                action_7d = np.zeros(7, dtype=np.float32)
                action_7d[:6] = raw_action
            elif action_dim == 8:
                # Drawer RL: 7 joint pos + 1 gripper → store as 7D (last joint + gripper)
                action_7d = np.zeros(7, dtype=np.float32)
                action_7d[:6] = raw_action[:6]
                action_7d[6] = raw_action[7] if len(raw_action) > 7 else 0.0
            else:
                action_7d = raw_action
                # Normalize gripper to 0-1 range (currently -1/+1)
                action_7d[6] = 0.0 if action_7d[6] < 0 else 1.0
            ep_actions.append(action_7d)

            # Timestamp
            ep_timestamps.append(np.float32(step / fps))

            # Physics GT: EE (position, orientation, velocity, angular velocity)
            ee_pos_np = gt["ee_position"][0].cpu().numpy()
            ee_vel_np = gt["ee_velocity"][0].cpu().numpy()
            ep_physics_gt["physics_gt.ee_position"].append(ee_pos_np)
            ep_physics_gt["physics_gt.ee_velocity"].append(ee_vel_np)
            # EE orientation (quaternion) and angular velocity
            if "ee_orientation" in gt:
                ep_physics_gt["physics_gt.ee_orientation"].append(gt["ee_orientation"][0].cpu().numpy())
            else:
                # Fallback: read from robot body data
                _hand_idx = robot.body_names.index("panda_hand")
                ep_physics_gt["physics_gt.ee_orientation"].append(robot.data.body_quat_w[0, _hand_idx].cpu().numpy())
            if "ee_angular_velocity" in gt:
                ep_physics_gt["physics_gt.ee_angular_velocity"].append(gt["ee_angular_velocity"][0].cpu().numpy())
            else:
                _hand_idx = robot.body_names.index("panda_hand")
                ep_physics_gt["physics_gt.ee_angular_velocity"].append(robot.data.body_ang_vel_w[0, _hand_idx].cpu().numpy())

            # EE acceleration via finite difference
            if prev_ee_vel is None:
                prev_ee_vel = ee_vel_np.copy()
            ee_accel = (ee_vel_np - prev_ee_vel) / dt
            ep_physics_gt["physics_gt.ee_acceleration"].append(ee_accel.astype(np.float32))
            prev_ee_vel = ee_vel_np.copy()

            # Handle different observation keys for different tasks
            if task_name == "reach":
                # Reach: no object, no contact — only EE + target
                target_pos = gt["target_position"][0, :3].cpu().numpy()
                ep_physics_gt["physics_gt.target_position"].append(target_pos.astype(np.float32))
                ee_target_dist = float(np.linalg.norm(ee_pos_np - target_pos))
                ep_physics_gt["physics_gt.ee_to_target_distance"].append(np.array([ee_target_dist], dtype=np.float32))

            elif task_name == "peg_insert":
                peg_pos = gt["peg_position"][0].cpu().numpy()
                hole_pos = gt["hole_position"][0].cpu().numpy()
                ep_physics_gt["physics_gt.peg_position"].append(peg_pos)
                ep_physics_gt["physics_gt.peg_orientation"].append(gt["peg_orientation"][0].cpu().numpy())
                ep_physics_gt["physics_gt.peg_velocity"].append(gt["peg_velocity"][0].cpu().numpy())
                ep_physics_gt["physics_gt.peg_angular_velocity"].append(gt["peg_angular_velocity"][0].cpu().numpy())
                ep_physics_gt["physics_gt.hole_position"].append(hole_pos)
                # Pair-specific contacts
                ep_physics_gt["physics_gt.contact_finger_l_peg_flag"].append(gt["finger_l_contact_flag"][0].cpu().numpy())
                ep_physics_gt["physics_gt.contact_finger_l_peg_force"].append(gt["finger_l_contact_force"][0].cpu().numpy())
                ep_physics_gt["physics_gt.contact_finger_r_peg_flag"].append(gt["finger_r_contact_flag"][0].cpu().numpy())
                ep_physics_gt["physics_gt.contact_finger_r_peg_force"].append(gt["finger_r_contact_force"][0].cpu().numpy())
                ep_physics_gt["physics_gt.contact_peg_socket_flag"].append(gt["held_fixed_contact_flag"][0].cpu().numpy())
                ep_physics_gt["physics_gt.contact_peg_socket_force"].append(gt["held_fixed_contact_force"][0].cpu().numpy())
                # Task-specific raw
                # insertion_depth: positive = peg above hole, zero/negative = inserted
                # peg descends from above, so peg_z > hole_z before insertion
                insertion_depth = float(peg_pos[2] - hole_pos[2])
                lateral_error = float(np.linalg.norm(peg_pos[:2] - hole_pos[:2]))
                ep_physics_gt["physics_gt.insertion_depth"].append(np.array([insertion_depth], dtype=np.float32))
                ep_physics_gt["physics_gt.peg_hole_lateral_error"].append(np.array([lateral_error], dtype=np.float32))

            elif task_name == "nut_thread":
                nut_pos = gt["nut_position"][0].cpu().numpy()
                nut_quat = gt["nut_orientation"][0].cpu().numpy()
                bolt_pos = gt["bolt_position"][0].cpu().numpy()
                bolt_quat = gt["bolt_orientation"][0].cpu().numpy()
                ep_physics_gt["physics_gt.nut_position"].append(nut_pos)
                ep_physics_gt["physics_gt.nut_orientation"].append(nut_quat)
                ep_physics_gt["physics_gt.nut_velocity"].append(gt["nut_velocity"][0].cpu().numpy())
                ep_physics_gt["physics_gt.nut_angular_velocity"].append(gt["nut_angular_velocity"][0].cpu().numpy())
                ep_physics_gt["physics_gt.bolt_position"].append(bolt_pos)
                ep_physics_gt["physics_gt.bolt_orientation"].append(bolt_quat)
                # Pair-specific contacts
                ep_physics_gt["physics_gt.contact_finger_l_nut_flag"].append(gt["finger_l_contact_flag"][0].cpu().numpy())
                ep_physics_gt["physics_gt.contact_finger_l_nut_force"].append(gt["finger_l_contact_force"][0].cpu().numpy())
                ep_physics_gt["physics_gt.contact_finger_r_nut_flag"].append(gt["finger_r_contact_flag"][0].cpu().numpy())
                ep_physics_gt["physics_gt.contact_finger_r_nut_force"].append(gt["finger_r_contact_force"][0].cpu().numpy())
                ep_physics_gt["physics_gt.contact_nut_bolt_flag"].append(gt["held_fixed_contact_flag"][0].cpu().numpy())
                ep_physics_gt["physics_gt.contact_nut_bolt_force"].append(gt["held_fixed_contact_force"][0].cpu().numpy())
                # Task-specific raw
                axial_progress = float(nut_pos[2] - bolt_pos[2])
                ep_physics_gt["physics_gt.axial_progress"].append(np.array([axial_progress], dtype=np.float32))
                # Nut-bolt relative angle: project onto bolt z-axis
                from scipy.spatial.transform import Rotation
                nut_r = Rotation.from_quat([nut_quat[1], nut_quat[2], nut_quat[3], nut_quat[0]])
                bolt_r = Rotation.from_quat([bolt_quat[1], bolt_quat[2], bolt_quat[3], bolt_quat[0]])
                rel_r = bolt_r.inv() * nut_r
                rel_angle = float(rel_r.as_rotvec()[2])  # z-axis component
                ep_physics_gt["physics_gt.nut_bolt_relative_angle"].append(np.array([rel_angle], dtype=np.float32))

            elif task_name == "drawer":
                jpos = gt["drawer_joint_pos"][0].cpu().numpy()
                ep_physics_gt["physics_gt.drawer_joint_pos"].append(jpos)
                ep_physics_gt["physics_gt.drawer_joint_vel"].append(gt["drawer_joint_vel"][0].cpu().numpy())
                if "handle_position" in gt:
                    ep_physics_gt["physics_gt.handle_position"].append(gt["handle_position"][0].cpu().numpy())
                    ep_physics_gt["physics_gt.handle_velocity"].append(gt["handle_velocity"][0].cpu().numpy())
                else:
                    ep_physics_gt["physics_gt.handle_position"].append(np.zeros(3, dtype=np.float32))
                    ep_physics_gt["physics_gt.handle_velocity"].append(np.zeros(3, dtype=np.float32))
                # Pair-specific: EE ↔ handle (from existing contact_sensor)
                if "contact_sensor" in env.scene.sensors:
                    sensor = env.scene.sensors["contact_sensor"]
                    ee_handle_force = sensor.data.force_matrix_w[0, 0, 0, :].cpu().numpy()
                    ee_handle_flag = 1.0 if np.linalg.norm(ee_handle_force) > 0.5 else 0.0
                    ep_physics_gt["physics_gt.contact_finger_l_handle_flag"].append(np.array([ee_handle_flag], dtype=np.float32))
                    ep_physics_gt["physics_gt.contact_finger_l_handle_force"].append(ee_handle_force.astype(np.float32))
                else:
                    ep_physics_gt["physics_gt.contact_finger_l_handle_flag"].append(np.zeros(1, dtype=np.float32))
                    ep_physics_gt["physics_gt.contact_finger_l_handle_force"].append(np.zeros(3, dtype=np.float32))
                # Right finger ↔ handle
                if "contact_sensor_r" in env.scene.sensors:
                    sensor_r = env.scene.sensors["contact_sensor_r"]
                    r_force = sensor_r.data.force_matrix_w[0, 0, 0, :].cpu().numpy()
                    r_flag = 1.0 if np.linalg.norm(r_force) > 0.5 else 0.0
                    ep_physics_gt["physics_gt.contact_finger_r_handle_flag"].append(np.array([r_flag], dtype=np.float32))
                    ep_physics_gt["physics_gt.contact_finger_r_handle_force"].append(r_force.astype(np.float32))
                else:
                    ep_physics_gt["physics_gt.contact_finger_r_handle_flag"].append(np.zeros(1, dtype=np.float32))
                    ep_physics_gt["physics_gt.contact_finger_r_handle_force"].append(np.zeros(3, dtype=np.float32))
                # Task-specific: drawer opening extent (normalized 0~1, max ~0.39m)
                drawer_max = 0.39
                opening_extent = float(np.clip(jpos[0] / drawer_max, 0.0, 1.0))
                ep_physics_gt["physics_gt.drawer_opening_extent"].append(np.array([opening_extent], dtype=np.float32))

            elif "object_position" in gt:
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
                obj_half = BALL_RADIUS if task_name == "strike" else CUBE_HALF_SIZE
                on_surface = 1.0 if obj_z < (TABLE_SURFACE_Z + obj_half + ON_SURFACE_MARGIN) else 0.0
                ep_physics_gt["physics_gt.object_on_surface"].append(np.array([on_surface], dtype=np.float32))

                # Pair-specific: EE ↔ object (existing contact_sensor = leftfinger → Object)
                if "contact_sensor" in env.scene.sensors:
                    sensor = env.scene.sensors["contact_sensor"]
                    ee_obj_force = sensor.data.force_matrix_w[0, 0, 0, :].cpu().numpy()
                    ee_obj_flag = 1.0 if np.linalg.norm(ee_obj_force) > 0.5 else 0.0
                else:
                    ee_obj_force = np.zeros(3, dtype=np.float32)
                    ee_obj_flag = 0.0
                ep_physics_gt["physics_gt.contact_finger_l_object_flag"].append(np.array([ee_obj_flag], dtype=np.float32))
                ep_physics_gt["physics_gt.contact_finger_l_object_force"].append(ee_obj_force.astype(np.float32))

                # Pair-specific: object ↔ surface (push/strike only)
                if task_name in ("push", "strike") and "object_surface_contact" in env.scene.sensors:
                    surf_sensor = env.scene.sensors["object_surface_contact"]
                    obj_surf_force = surf_sensor.data.force_matrix_w[0, 0, 0, :].cpu().numpy()
                    obj_surf_flag = 1.0 if np.linalg.norm(obj_surf_force) > 0.5 else 0.0
                    ep_physics_gt["physics_gt.contact_object_surface_flag"].append(np.array([obj_surf_flag], dtype=np.float32))
                    ep_physics_gt["physics_gt.contact_object_surface_force"].append(obj_surf_force.astype(np.float32))
                elif task_name in ("push", "strike"):
                    ep_physics_gt["physics_gt.contact_object_surface_flag"].append(np.zeros(1, dtype=np.float32))
                    ep_physics_gt["physics_gt.contact_object_surface_force"].append(np.zeros(3, dtype=np.float32))

                # Task-specific: object_to_target_distance (push/strike)
                if task_name in ("push", "strike"):
                    if "target_position" in gt:
                        target_np = gt["target_position"][0, :3].cpu().numpy()
                        # Check if target is valid (not zeros fallback)
                        if np.linalg.norm(target_np) > 0.01:
                            obj_target_dist = float(np.linalg.norm(obj_pos_np[:2] - target_np[:2]))
                        else:
                            obj_target_dist = float("nan")
                    else:
                        obj_target_dist = float("nan")
                    ep_physics_gt["physics_gt.object_to_target_distance"].append(np.array([obj_target_dist], dtype=np.float32))

                # Task-specific: ball_planar_travel_distance (strike — cumulative)
                if task_name == "strike":
                    if prev_ball_pos is None:
                        prev_ball_pos = obj_pos_np.copy()
                    ball_rolling_dist += float(np.linalg.norm(obj_pos_np[:2] - prev_ball_pos[:2]))
                    prev_ball_pos = obj_pos_np.copy()
                    ep_physics_gt["physics_gt.ball_planar_travel_distance"].append(np.array([ball_rolling_dist], dtype=np.float32))

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

                # Pair-specific: finger_l ↔ object (stack uses same contact_sensor)
                if "contact_sensor" in env.scene.sensors:
                    sensor = env.scene.sensors["contact_sensor"]
                    ee_obj_force = sensor.data.force_matrix_w[0, 0, 0, :].cpu().numpy()
                    ee_obj_flag = 1.0 if np.linalg.norm(ee_obj_force) > 0.5 else 0.0
                else:
                    ee_obj_force = np.zeros(3, dtype=np.float32)
                    ee_obj_flag = 0.0
                ep_physics_gt["physics_gt.contact_finger_l_object_flag"].append(np.array([ee_obj_flag], dtype=np.float32))
                ep_physics_gt["physics_gt.contact_finger_l_object_force"].append(ee_obj_force.astype(np.float32))

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

            # Gripper contact (not for reach)
            if task_name not in ("reach",):
                ep_physics_gt["physics_gt.contact_flag"].append(gt["contact_flag"][0].cpu().numpy())
                ep_physics_gt["physics_gt.contact_force"].append(gt["contact_force"][0].cpu().numpy())

                if "contact_sensor" in env.scene.sensors:
                    sensor = env.scene.sensors["contact_sensor"]
                    contact_pos = sensor.data.contact_pos_w[0, 0, 0, :].cpu().numpy()
                    contact_pos = np.nan_to_num(contact_pos, nan=0.0)
                    ep_physics_gt["physics_gt.contact_point"].append(contact_pos.astype(np.float32))
                else:
                    ep_physics_gt["physics_gt.contact_point"].append(np.zeros(3, dtype=np.float32))

            # Phase label
            if oracle_policy is not None:
                state_idx = oracle_policy.state[0].item()
                phase_map = TASK_STATE_TO_PHASE.get(task_name, {})
                phase = phase_map.get(state_idx, 7)
            else:
                phase = 7  # idle for random actions
            ep_physics_gt["physics_gt.phase"].append(np.array([phase], dtype=np.float32))

            # Target position for pick_place/push (Critical fix #2)
            if task_name in ("pick_place", "push", "strike") and "target_position" in gt:
                target_3d = gt["target_position"][0, :3].cpu().numpy()
                ep_physics_gt["physics_gt.target_position"].append(target_3d.astype(np.float32))

            # Camera frames
            if is_factory:
                table_rgb, wrist_rgb = env.get_camera_data()
                ep_frames_table.append(table_rgb[0].cpu().numpy().astype(np.uint8))
                ep_frames_wrist.append(wrist_rgb[0].cpu().numpy().astype(np.uint8))
            elif "table_cam" in env.scene.sensors:
                # Read from sensors directly (works for both dict and flat obs envs)
                t_rgb = env.scene.sensors["table_cam"].data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
                ep_frames_table.append(t_rgb)
                if "wrist_cam" in env.scene.sensors:
                    w_rgb = env.scene.sensors["wrist_cam"].data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
                    ep_frames_wrist.append(w_rgb)
            else:
                policy_obs = obs["policy"]
                if isinstance(policy_obs, dict):
                    if "table_cam" in policy_obs:
                        ep_frames_table.append(policy_obs["table_cam"][0].cpu().numpy())
                    if "wrist_cam" in policy_obs:
                        ep_frames_wrist.append(policy_obs["wrist_cam"][0].cpu().numpy())

            # Step environment
            if is_rl_wrapped:
                # Drawer RL: step through wrapped env for policy obs
                obs_wrapped, rew_wrapped, dones, infos = _drawer_env_wrapped.step(action)
                obs = {"policy": obs_wrapped}  # wrap for compatibility
                reward = rew_wrapped
                terminated = dones
                truncated = dones
                info = infos
            else:
                obs, reward, terminated, truncated, info = env.step(action)

            # Record reward (Critical fix #4: for next.done)
            ep_rewards.append(reward[0].item())

            # Print progress
            if step % 100 == 0:
                print(f"  Step {step}/{max_steps}, reward={reward[0].item():.4f}")

        ep_length = len(ep_states)
        print(f"  Episode {ep_idx} complete: {ep_length} steps")

        # Validate physics GT lengths match ep_length
        for k in physics_gt_keys:
            actual = len(ep_physics_gt[k])
            assert actual == ep_length, (
                f"physics_gt key length mismatch: {k} has {actual} entries, expected {ep_length}"
            )

        # --- Success判定 (before save) ---
        success = False
        try:
            if task_name in ("push", "strike") and "physics_gt.object_position" in ep_physics_gt and "physics_gt.target_position" in ep_physics_gt:
                final_obj = np.array(ep_physics_gt["physics_gt.object_position"][-1])
                final_target = np.array(ep_physics_gt["physics_gt.target_position"][-1])
                success = float(np.linalg.norm(final_obj[:2] - final_target[:2])) < 0.06
            elif task_name in ("peg_insert", "nut_thread"):
                # Factory tasks: max reward > 1.0 = success
                # (per-step reward can drop at truncation, so use max over episode)
                max_reward = max(ep_rewards) if ep_rewards else 0.0
                success = max_reward > 1.0
            elif task_name == "drawer" and "physics_gt.drawer_joint_pos" in ep_physics_gt:
                success = float(ep_physics_gt["physics_gt.drawer_joint_pos"][-1][0]) > 0.1
            elif task_name == "reach" and "physics_gt.ee_position" in ep_physics_gt and "physics_gt.target_position" in ep_physics_gt:
                final_ee = np.array(ep_physics_gt["physics_gt.ee_position"][-1])
                final_target = np.array(ep_physics_gt["physics_gt.target_position"][-1])
                success = float(np.linalg.norm(final_ee - final_target)) < 0.02
        except (IndexError, KeyError):
            success = False

        if filter_success and not success:
            print(f"  Episode FAILED — skipping (filter_success=True)")
            continue

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
            "success": bool(success),
        }
        ep_meta.update(physics_params)
        episodes_meta.append(ep_meta)

        # Accumulate stats
        all_episode_stats["observation.state"].extend(ep_states)
        all_episode_stats["action"].extend(ep_actions)
        all_episode_stats["timestamp"].extend(ep_timestamps)
        all_episodes_physics_gt.append({k: list(v) for k, v in ep_physics_gt.items()})

        global_index += ep_length
        saved_episodes += 1

    if filter_success:
        print(f"\nSuccess filter: {saved_episodes}/{attempted_episodes} episodes saved ({saved_episodes/max(attempted_episodes,1)*100:.0f}% success rate)")

    # --- Save metadata ---
    meta_dir = os.path.join(output_dir, "meta")
    os.makedirs(meta_dir, exist_ok=True)

    # info.json
    total_frames = global_index

    # Build physics GT shapes dict — must match physics_gt_keys
    _shape_map = {
        # Common EE
        "physics_gt.ee_position": [3],
        "physics_gt.ee_orientation": [4],
        "physics_gt.ee_velocity": [3],
        "physics_gt.ee_angular_velocity": [3],
        "physics_gt.ee_acceleration": [3],
        # Object (push/strike/lift/pick_place/stack)
        "physics_gt.object_position": [3],
        "physics_gt.object_orientation": [4],
        "physics_gt.object_velocity": [3],
        "physics_gt.object_angular_velocity": [3],
        "physics_gt.object_acceleration": [3],
        "physics_gt.ee_to_object_distance": [1],
        "physics_gt.object_on_surface": [1],
        # Legacy single contact
        "physics_gt.contact_flag": [1],
        "physics_gt.contact_force": [3],
        "physics_gt.contact_point": [3],
        "physics_gt.phase": [1],
        "physics_gt.target_position": [3],
        "physics_gt.ee_to_target_distance": [1],
        # Pair-specific: EE ↔ object (push/strike/lift/pick_place/stack)
        "physics_gt.contact_finger_l_object_flag": [1],
        "physics_gt.contact_finger_l_object_force": [3],
        # Pair-specific: object ↔ surface (push/strike)
        "physics_gt.contact_object_surface_flag": [1],
        "physics_gt.contact_object_surface_force": [3],
        # Task-specific: push/strike
        "physics_gt.object_to_target_distance": [1],
        "physics_gt.ball_planar_travel_distance": [1],
        # Drawer
        "physics_gt.drawer_joint_pos": [1],
        "physics_gt.drawer_joint_vel": [1],
        "physics_gt.handle_position": [3],
        "physics_gt.handle_velocity": [3],
        "physics_gt.contact_finger_l_handle_flag": [1],
        "physics_gt.contact_finger_l_handle_force": [3],
        "physics_gt.contact_finger_r_handle_flag": [1],
        "physics_gt.contact_finger_r_handle_force": [3],
        "physics_gt.drawer_opening_extent": [1],
        # Peg insert
        "physics_gt.peg_position": [3],
        "physics_gt.peg_orientation": [4],
        "physics_gt.peg_velocity": [3],
        "physics_gt.peg_angular_velocity": [3],
        "physics_gt.hole_position": [3],
        "physics_gt.contact_finger_l_peg_flag": [1],
        "physics_gt.contact_finger_l_peg_force": [3],
        "physics_gt.contact_finger_r_peg_flag": [1],
        "physics_gt.contact_finger_r_peg_force": [3],
        "physics_gt.contact_peg_socket_flag": [1],
        "physics_gt.contact_peg_socket_force": [3],
        "physics_gt.insertion_depth": [1],
        "physics_gt.peg_hole_lateral_error": [1],
        # Nut thread
        "physics_gt.nut_position": [3],
        "physics_gt.nut_orientation": [4],
        "physics_gt.nut_velocity": [3],
        "physics_gt.nut_angular_velocity": [3],
        "physics_gt.bolt_position": [3],
        "physics_gt.bolt_orientation": [4],
        "physics_gt.contact_finger_l_nut_flag": [1],
        "physics_gt.contact_finger_l_nut_force": [3],
        "physics_gt.contact_finger_r_nut_flag": [1],
        "physics_gt.contact_finger_r_nut_force": [3],
        "physics_gt.contact_nut_bolt_flag": [1],
        "physics_gt.contact_nut_bolt_force": [3],
        "physics_gt.axial_progress": [1],
        "physics_gt.nut_bolt_relative_angle": [1],
        # Stack
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
    }
    physics_gt_shapes = {k: _shape_map[k] for k in physics_gt_keys}

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
    print(f"  Episodes: {saved_episodes}/{num_episodes} (attempted: {attempted_episodes})")
    print(f"  Total frames: {total_frames}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")

    env.close()


def collect_task_parallel(task_name: str, num_episodes: int, num_envs: int, output_dir: str,
                           rl_checkpoint: str | None = None, filter_success: bool = False):
    """Parallel data collection using multiple envs. Designed for Factory tasks (peg_insert/nut_thread)."""
    import json
    from collections import defaultdict
    from scipy.spatial.transform import Rotation

    is_factory = task_name in ("peg_insert", "nut_thread")

    # --- Setup env ---
    if is_factory:
        from physrepa_tasks.envs.factory_camera_env import (
            FactoryCameraEnv, PegInsertCameraCfg, NutThreadCameraCfg,
        )
        cfg = PegInsertCameraCfg() if task_name == "peg_insert" else NutThreadCameraCfg()
        cfg.scene.num_envs = num_envs
        cfg.scene.env_spacing = 20.0  # large spacing so cameras don't see neighboring envs
        if task_name == "nut_thread":
            cfg.episode_length_s = 10.0  # override 30s → 10s (sufficient for threading)
        env = FactoryCameraEnv(cfg=cfg)
    else:
        raise ValueError(f"Parallel collection only supports Factory tasks, got {task_name}")

    # Load RL policy
    if rl_checkpoint is not None:
        from physrepa_tasks.utils.rl_games_policy import RlGamesPolicy
        rl_policy = RlGamesPolicy(rl_checkpoint, device=env.device)
        print(f"  Loaded RL-Games checkpoint: {rl_checkpoint}")
    else:
        rl_policy = None

    dt = cfg.sim.dt * cfg.decimation
    fps = int(1.0 / dt)
    max_steps = int(cfg.episode_length_s / dt)

    print(f"\n{'='*60}")
    print(f"Parallel collecting {num_episodes} episodes for task: {task_name}")
    print(f"  num_envs={num_envs}, max_steps={max_steps}, dt={dt:.4f}, fps={fps}")
    print(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)
    CHUNKS_SIZE = 1000

    # --- Per-env episode buffers ---
    class EpBuffer:
        def __init__(self):
            self.reset()

        def reset(self):
            self.states = []
            self.actions = []
            self.rewards = []
            self.timestamps = []
            self.frames_table = []
            self.frames_wrist = []
            self.physics_gt = defaultdict(list)
            self.step_count = 0
            self.physics_params = {}

    buffers = [EpBuffer() for _ in range(num_envs)]

    # --- Task metadata ---
    task_desc = TASK_INSTRUCTION_TEMPLATES[task_name]
    saved_count = 0
    attempted_count = 0
    global_index = 0
    episodes_meta = []
    all_episode_stats = {"observation.state": [], "action": [], "timestamp": []}
    all_episodes_physics_gt = []

    # --- Reset all envs ---
    obs, info = env.reset()

    # Render warmup (flush stale camera frames without consuming episode budget)
    for _ in range(3):
        env.sim.render()

    # Randomize physics for all envs
    if hasattr(env, 'randomize_physics'):
        env.randomize_physics()

    # Reset RL policy
    if rl_policy is not None:
        rl_policy.reset()

    # Read initial physics params (same for all envs in this batch — simplified)
    physics_params = read_physics_params(env, task_name)

    for i in range(num_envs):
        buffers[i].physics_params = dict(physics_params)

    print(f"  Physics params: {physics_params}")

    step = 0
    while saved_count < num_episodes:
        # --- Generate actions ---
        if rl_policy is not None:
            with torch.inference_mode():
                flat_obs = obs["policy"] if isinstance(obs, dict) else obs
                action = rl_policy(flat_obs)  # (num_envs, 6)
                if action.dim() == 1:
                    action = action.unsqueeze(0)
        else:
            action = torch.zeros(num_envs, 6, device=env.device)

        # --- Record pre-step data for all envs ---
        phys = env.get_physics_data()
        table_rgb, wrist_rgb = env.get_camera_data()

        for i in range(num_envs):
            buf = buffers[i]
            if buf.step_count >= max_steps:
                continue  # this env's episode is done, waiting for save

            # EE state → 8D
            ee_pos = phys["ee_position"][i].cpu().numpy()
            ee_quat = phys["ee_orientation"][i].cpu().numpy()
            from scipy.spatial.transform import Rotation as R
            quat_xyzw = np.array([ee_quat[1], ee_quat[2], ee_quat[3], ee_quat[0]])
            rpy = R.from_quat(quat_xyzw).as_euler("xyz")
            gripper_val = env.joint_pos[i, -1].item() / 0.04
            state_8d = np.array([ee_pos[0], ee_pos[1], ee_pos[2], rpy[0], rpy[1], rpy[2], 0.0, gripper_val], dtype=np.float32)
            buf.states.append(state_8d)

            # Action → 7D
            raw_action = action[i].cpu().numpy().astype(np.float32)
            action_7d = np.zeros(7, dtype=np.float32)
            action_7d[:6] = raw_action[:6]
            buf.actions.append(action_7d)

            # Timestamp
            buf.timestamps.append(np.float32(buf.step_count / fps))

            # Camera frames
            buf.frames_table.append(table_rgb[i].cpu().numpy().astype(np.uint8))
            buf.frames_wrist.append(wrist_rgb[i].cpu().numpy().astype(np.uint8))

            # Physics GT: EE
            ee_vel = phys["ee_velocity"][i].cpu().numpy()
            buf.physics_gt["physics_gt.ee_position"].append(ee_pos)
            buf.physics_gt["physics_gt.ee_orientation"].append(ee_quat)
            buf.physics_gt["physics_gt.ee_velocity"].append(ee_vel)
            buf.physics_gt["physics_gt.ee_angular_velocity"].append(phys["ee_angular_velocity"][i].cpu().numpy())

            # EE acceleration
            if len(buf.physics_gt["physics_gt.ee_velocity"]) < 2:
                buf.physics_gt["physics_gt.ee_acceleration"].append(np.zeros(3, dtype=np.float32))
            else:
                prev_vel = buf.physics_gt["physics_gt.ee_velocity"][-2]
                buf.physics_gt["physics_gt.ee_acceleration"].append(((ee_vel - prev_vel) / dt).astype(np.float32))

            # Task-specific GT
            if task_name == "peg_insert":
                peg_pos = phys["held_position"][i].cpu().numpy()
                hole_pos = phys["fixed_position"][i].cpu().numpy()
                buf.physics_gt["physics_gt.peg_position"].append(peg_pos)
                buf.physics_gt["physics_gt.peg_orientation"].append(phys["held_orientation"][i].cpu().numpy())
                buf.physics_gt["physics_gt.peg_velocity"].append(phys["held_velocity"][i].cpu().numpy())
                buf.physics_gt["physics_gt.peg_angular_velocity"].append(phys["held_angular_velocity"][i].cpu().numpy())
                buf.physics_gt["physics_gt.hole_position"].append(hole_pos)
                # Contacts (zeros — ArticulationView limitation)
                buf.physics_gt["physics_gt.contact_flag"].append(phys["contact_flag"][i].cpu().numpy())
                buf.physics_gt["physics_gt.contact_force"].append(phys["contact_force"][i].cpu().numpy())
                buf.physics_gt["physics_gt.contact_point"].append(np.zeros(3, dtype=np.float32))
                buf.physics_gt["physics_gt.contact_finger_l_peg_flag"].append(phys["finger_l_contact_flag"][i].cpu().numpy())
                buf.physics_gt["physics_gt.contact_finger_l_peg_force"].append(phys["finger_l_contact_force"][i].cpu().numpy())
                buf.physics_gt["physics_gt.contact_finger_r_peg_flag"].append(phys["finger_r_contact_flag"][i].cpu().numpy())
                buf.physics_gt["physics_gt.contact_finger_r_peg_force"].append(phys["finger_r_contact_force"][i].cpu().numpy())
                buf.physics_gt["physics_gt.contact_peg_socket_flag"].append(phys["held_fixed_contact_flag"][i].cpu().numpy())
                buf.physics_gt["physics_gt.contact_peg_socket_force"].append(phys["held_fixed_contact_force"][i].cpu().numpy())
                # Task-specific raw
                insertion_depth = float(peg_pos[2] - hole_pos[2])
                lateral_error = float(np.linalg.norm(peg_pos[:2] - hole_pos[:2]))
                buf.physics_gt["physics_gt.insertion_depth"].append(np.array([insertion_depth], dtype=np.float32))
                buf.physics_gt["physics_gt.peg_hole_lateral_error"].append(np.array([lateral_error], dtype=np.float32))
            elif task_name == "nut_thread":
                nut_pos = phys["held_position"][i].cpu().numpy()
                nut_quat = phys["held_orientation"][i].cpu().numpy()
                bolt_pos = phys["fixed_position"][i].cpu().numpy()
                bolt_quat = phys["fixed_orientation"][i].cpu().numpy()
                buf.physics_gt["physics_gt.nut_position"].append(nut_pos)
                buf.physics_gt["physics_gt.nut_orientation"].append(nut_quat)
                buf.physics_gt["physics_gt.nut_velocity"].append(phys["held_velocity"][i].cpu().numpy())
                buf.physics_gt["physics_gt.nut_angular_velocity"].append(phys["held_angular_velocity"][i].cpu().numpy())
                buf.physics_gt["physics_gt.bolt_position"].append(bolt_pos)
                buf.physics_gt["physics_gt.bolt_orientation"].append(bolt_quat)
                buf.physics_gt["physics_gt.contact_flag"].append(phys["contact_flag"][i].cpu().numpy())
                buf.physics_gt["physics_gt.contact_force"].append(phys["contact_force"][i].cpu().numpy())
                buf.physics_gt["physics_gt.contact_point"].append(np.zeros(3, dtype=np.float32))
                buf.physics_gt["physics_gt.contact_finger_l_nut_flag"].append(phys["finger_l_contact_flag"][i].cpu().numpy())
                buf.physics_gt["physics_gt.contact_finger_l_nut_force"].append(phys["finger_l_contact_force"][i].cpu().numpy())
                buf.physics_gt["physics_gt.contact_finger_r_nut_flag"].append(phys["finger_r_contact_flag"][i].cpu().numpy())
                buf.physics_gt["physics_gt.contact_finger_r_nut_force"].append(phys["finger_r_contact_force"][i].cpu().numpy())
                buf.physics_gt["physics_gt.contact_nut_bolt_flag"].append(phys["held_fixed_contact_flag"][i].cpu().numpy())
                buf.physics_gt["physics_gt.contact_nut_bolt_force"].append(phys["held_fixed_contact_force"][i].cpu().numpy())
                axial_progress = float(nut_pos[2] - bolt_pos[2])
                buf.physics_gt["physics_gt.axial_progress"].append(np.array([axial_progress], dtype=np.float32))
                nut_r = Rotation.from_quat([nut_quat[1], nut_quat[2], nut_quat[3], nut_quat[0]])
                bolt_r = Rotation.from_quat([bolt_quat[1], bolt_quat[2], bolt_quat[3], bolt_quat[0]])
                rel_angle = float((bolt_r.inv() * nut_r).as_rotvec()[2])
                buf.physics_gt["physics_gt.nut_bolt_relative_angle"].append(np.array([rel_angle], dtype=np.float32))

            # Phase (RL = idle)
            buf.physics_gt["physics_gt.phase"].append(np.array([7], dtype=np.float32))
            buf.step_count += 1

        # --- Step all envs ---
        obs, reward, terminated, truncated, info = env.step(action)

        # Record rewards
        for i in range(num_envs):
            if buffers[i].step_count <= max_steps:
                buffers[i].rewards.append(reward[i].item())

        # --- Check for completed episodes ---
        dones = terminated | truncated  # (num_envs,)
        done_envs = []
        for i in range(num_envs):
            if buffers[i].step_count >= max_steps or (dones[i] if dones.dim() > 0 else dones.item()):
                done_envs.append(i)

        for i in done_envs:
            buf = buffers[i]
            ep_length = len(buf.states)
            if ep_length == 0:
                buf.reset()
                continue

            attempted_count += 1

            # Success check
            max_reward = max(buf.rewards) if buf.rewards else 0.0
            success = max_reward > 1.0

            if filter_success and not success:
                print(f"  Env {i}: Episode FAILED (max_rew={max_reward:.2f}) — skipping")
                buf.reset()
                if rl_policy is not None:
                    rl_policy.reset(env_ids=[i])
                continue

            ep_idx = saved_count
            print(f"  Env {i}: Episode {ep_idx} complete: {ep_length} steps (max_rew={max_reward:.2f})")

            # --- Save parquet ---
            chunk_idx = ep_idx // CHUNKS_SIZE
            chunk_dir = f"chunk-{chunk_idx:03d}"
            parquet_dir = os.path.join(output_dir, "data", chunk_dir)
            os.makedirs(parquet_dir, exist_ok=True)

            physics_gt_keys = list(buf.physics_gt.keys())
            df_data = {
                "observation.state": [s.tolist() for s in buf.states],
                "action": [a.tolist() for a in buf.actions],
                "timestamp": buf.timestamps,
                "frame_index": list(range(ep_length)),
                "episode_index": [ep_idx] * ep_length,
                "index": list(range(global_index, global_index + ep_length)),
                "task_index": [0] * ep_length,
                "next.done": [False] * (ep_length - 1) + [True],
                "next.reward": buf.rewards[:ep_length],
            }
            for gt_key in physics_gt_keys:
                df_data[gt_key] = [v.tolist() for v in buf.physics_gt[gt_key]]

            df = pd.DataFrame(df_data)
            parquet_path = os.path.join(parquet_dir, f"episode_{ep_idx:06d}.parquet")
            df.to_parquet(parquet_path, index=False)

            # --- Save videos ---
            if buf.frames_table:
                video_dir = os.path.join(output_dir, "videos", chunk_dir, "observation.images.image_0")
                encode_video(buf.frames_table, os.path.join(video_dir, f"episode_{ep_idx:06d}.mp4"), fps=fps)
            if buf.frames_wrist:
                video_dir = os.path.join(output_dir, "videos", chunk_dir, "observation.images.image_1")
                encode_video(buf.frames_wrist, os.path.join(video_dir, f"episode_{ep_idx:06d}.mp4"), fps=fps)

            # --- Episode meta ---
            ep_meta = {
                "episode_index": ep_idx,
                "tasks": [task_desc],
                "length": ep_length,
                "success": bool(success),
            }
            ep_meta.update(buf.physics_params)
            episodes_meta.append(ep_meta)

            all_episode_stats["observation.state"].extend(buf.states)
            all_episode_stats["action"].extend(buf.actions)
            all_episode_stats["timestamp"].extend(buf.timestamps)
            all_episodes_physics_gt.append({k: list(v) for k, v in buf.physics_gt.items()})

            global_index += ep_length
            saved_count += 1

            if saved_count % 10 == 0:
                print(f"  >>> {saved_count}/{num_episodes} episodes saved ({attempted_count} attempted)")

            # Reset this env's buffer
            buf.reset()
            if rl_policy is not None:
                rl_policy.reset(env_ids=[i])

            # Re-randomize physics
            if hasattr(env, 'randomize_physics'):
                env.randomize_physics()
                buffers[i].physics_params = read_physics_params(env, task_name)

            if saved_count >= num_episodes:
                break

        step += 1

    # --- Save metadata ---
    meta_dir = os.path.join(output_dir, "meta")
    os.makedirs(meta_dir, exist_ok=True)

    info_dict = {
        "codebase_version": "v2.0",
        "robot_type": "franka",
        "total_episodes": saved_count,
        "total_frames": global_index,
        "total_tasks": 1,
        "total_videos": 2,
        "total_chunks": (saved_count // CHUNKS_SIZE) + 1,
        "chunks_size": CHUNKS_SIZE,
        "fps": fps,
        "splits": {"train": f"0:{saved_count}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    }
    with open(os.path.join(meta_dir, "info.json"), "w") as f:
        json.dump(info_dict, f, indent=2)

    with open(os.path.join(meta_dir, "episodes.jsonl"), "w") as f:
        for ep in episodes_meta:
            f.write(json.dumps(ep) + "\n")

    with open(os.path.join(meta_dir, "tasks.jsonl"), "w") as f:
        f.write(json.dumps({"task_index": 0, "task": task_desc}) + "\n")

    print(f"\n{'='*60}")
    print(f"Parallel collection complete for task: {task_name}")
    print(f"  Episodes: {saved_count}/{num_episodes} (attempted: {attempted_count})")
    print(f"  Total frames: {global_index}")
    print(f"  num_envs: {num_envs}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")

    env.close()


def main():
    output_dir = os.path.join(args_cli.output_dir, args_cli.task)
    if args_cli.num_envs > 1:
        collect_task_parallel(args_cli.task, args_cli.num_episodes, args_cli.num_envs, output_dir,
                              rl_checkpoint=args_cli.rl_checkpoint, filter_success=args_cli.filter_success)
    else:
        collect_task(args_cli.task, args_cli.num_episodes, output_dir,
                     use_oracle=args_cli.use_oracle, rl_checkpoint=args_cli.rl_checkpoint,
                     step0=args_cli.step0, filter_success=args_cli.filter_success)
    simulation_app.close()


if __name__ == "__main__":
    main()
