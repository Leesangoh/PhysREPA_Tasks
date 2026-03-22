"""Test all 4 PhysREPA environments with scripted oracle policies.

Saves sample images and prints physics GT for sanity checking.

Usage:
    cd /home/solee/IsaacLab
    PYTHONPATH=/home/solee:$PYTHONPATH ./isaaclab.sh -p /home/solee/physrepa_tasks/test_all_envs.py
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

# Force unbuffered output
import functools
print = functools.partial(print, flush=True)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test all PhysREPA environments")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments")
parser.add_argument("--task", type=str, default="all", choices=["all", "lift", "pick_place", "push", "stack"])
parser.add_argument("--output_dir", type=str, default="/mnt/md1/solee/data/isaac_physrepa/sanity_check")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Post-launch imports
import numpy as np
from PIL import Image

from isaaclab.envs import ManagerBasedRLEnv

from physrepa_tasks.envs.lift_env_cfg import PhysREPALiftEnvCfg
from physrepa_tasks.envs.pick_place_env_cfg import PhysREPAPickPlaceEnvCfg
from physrepa_tasks.envs.push_env_cfg import PhysREPAPushEnvCfg
from physrepa_tasks.envs.stack_env_cfg import PhysREPAStackEnvCfg
from physrepa_tasks.policies.scripted_policy import LiftPolicy, PickPlacePolicy, PushPolicy, StackPolicy


TASK_CONFIGS = {
    "lift": (PhysREPALiftEnvCfg, LiftPolicy),
    "pick_place": (PhysREPAPickPlaceEnvCfg, PickPlacePolicy),
    "push": (PhysREPAPushEnvCfg, PushPolicy),
    "stack": (PhysREPAStackEnvCfg, StackPolicy),
}


def save_image(tensor: torch.Tensor, path: str):
    """Save a single image tensor (H, W, 3) uint8 to file."""
    img = tensor.cpu().numpy()
    Image.fromarray(img).save(path)


def test_task(task_name: str, num_envs: int, output_dir: str):
    """Test a single task: create env, run policy, save images, print physics GT."""
    print(f"\n{'='*70}")
    print(f"TESTING TASK: {task_name}")
    print(f"{'='*70}")

    cfg_cls, policy_cls = TASK_CONFIGS[task_name]
    cfg = cfg_cls()
    cfg.scene.num_envs = num_envs

    env = ManagerBasedRLEnv(cfg=cfg)
    policy = policy_cls(num_envs=num_envs, device=env.device)

    task_dir = os.path.join(output_dir, task_name)
    os.makedirs(task_dir, exist_ok=True)

    print(f"  Env created: {num_envs} envs, device={env.device}")
    print(f"  Episode length: {cfg.episode_length_s}s")

    # Reset
    obs, info = env.reset()
    policy.reset()

    # Check observation structure
    print(f"\n--- Observation Structure ---")
    for group_name, group_obs in obs.items():
        print(f"  Group: {group_name}")
        if isinstance(group_obs, dict):
            for key, val in group_obs.items():
                if isinstance(val, torch.Tensor):
                    shape_str = f"shape={val.shape}, dtype={val.dtype}"
                    if val.numel() < 20:
                        print(f"    {key}: {shape_str}, val={val[0].tolist()}")
                    else:
                        print(f"    {key}: {shape_str}")

    # Save initial camera images
    if "policy" in obs and isinstance(obs["policy"], dict):
        for cam_name in ["table_cam", "wrist_cam"]:
            if cam_name in obs["policy"]:
                img = obs["policy"][cam_name]
                save_image(img[0], os.path.join(task_dir, f"{cam_name}_step000.png"))
                print(f"  Saved {cam_name}_step000.png (min={img.min()}, max={img.max()})")

    # Run episode with scripted policy
    total_steps = int(cfg.episode_length_s / (cfg.sim.dt * cfg.decimation))
    contact_detected = False
    print(f"\n--- Running {total_steps} steps with {policy_cls.__name__} ---")

    for step in range(total_steps):
        action = policy.get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        # Handle resets
        reset_ids = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1)
        if len(reset_ids) > 0:
            policy.reset(reset_ids)

        gt = obs.get("physics_gt", {})
        if isinstance(gt, dict):
            cf = gt.get("contact_flag", torch.zeros(1))
            if cf.max() > 0 and not contact_detected:
                contact_detected = True
                print(f"  [Step {step}] FIRST CONTACT DETECTED!")
                force = gt.get("contact_force", torch.zeros(1))
                print(f"    contact_flag={cf[0].item():.0f}, force_norm={torch.norm(force[0]).item():.3f}")
                if "ee_position" in gt:
                    print(f"    ee_pos={gt['ee_position'][0].tolist()}")
                obj_key = "object_position" if "object_position" in gt else "cube_a_position"
                if obj_key in gt:
                    print(f"    obj_pos={gt[obj_key][0].tolist()}")

        # Print state info at key intervals
        if step in [0, total_steps // 4, total_steps // 2, 3 * total_steps // 4, total_steps - 1]:
            state_val = policy.state[0].item()
            rew_val = reward[0].item() if reward.numel() > 0 else 0
            msg = f"  [Step {step:4d}] state={state_val}, reward={rew_val:.4f}"
            if isinstance(gt, dict):
                if "ee_position" in gt:
                    msg += f", ee_z={gt['ee_position'][0, 2].item():.4f}"
                obj_key = "object_position" if "object_position" in gt else "cube_a_position"
                if obj_key in gt:
                    msg += f", obj_z={gt[obj_key][0, 2].item():.4f}"
                if "contact_flag" in gt:
                    msg += f", contact={gt['contact_flag'][0].item():.0f}"
            print(msg)

        # Save mid and final frame
        if step == total_steps // 2 or step == total_steps - 1:
            if "policy" in obs and isinstance(obs["policy"], dict):
                for cam_name in ["table_cam", "wrist_cam"]:
                    if cam_name in obs["policy"]:
                        img = obs["policy"][cam_name]
                        save_image(img[0], os.path.join(task_dir, f"{cam_name}_step{step:03d}.png"))

    print(f"\n--- Summary for {task_name} ---")
    print(f"  Contact detected: {contact_detected}")
    print(f"  Final policy state (env 0): {policy.state[0].item()}")
    if isinstance(gt, dict):
        obj_key = "object_position" if "object_position" in gt else "cube_a_position"
        if obj_key in gt:
            print(f"  Final object position: {gt[obj_key][0].tolist()}")
    print(f"  Images saved to: {task_dir}")
    print(f"  TASK {task_name} TEST COMPLETE\n")

    env.close()
    return True


def main():
    os.makedirs(args_cli.output_dir, exist_ok=True)

    tasks = list(TASK_CONFIGS.keys()) if args_cli.task == "all" else [args_cli.task]

    results = {}
    for task_name in tasks:
        try:
            success = test_task(task_name, args_cli.num_envs, args_cli.output_dir)
            results[task_name] = "PASS" if success else "FAIL"
        except Exception as e:
            print(f"\n  ERROR in {task_name}: {e}")
            import traceback
            traceback.print_exc()
            results[task_name] = f"ERROR: {e}"

    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    for task, status in results.items():
        print(f"  {task:15s}: {status}")
    print(f"{'='*70}")

    simulation_app.close()


if __name__ == "__main__":
    main()
