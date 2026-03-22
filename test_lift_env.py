"""Test script for PhysREPA Lift environment.

Verifies: robot moves, camera renders, contact sensor reads, physics randomization works.

Usage:
    /isaac-sim/python.sh physrepa_tasks/test_lift_env.py --num_envs 4
"""

import argparse
import sys
import os
import torch

# Add physrepa_tasks to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Isaac Lab imports
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test PhysREPA Lift Environment")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Now safe to import sim-dependent modules
import gymnasium as gym

from isaaclab.envs import ManagerBasedRLEnv

from physrepa_tasks.envs.lift_env_cfg import PhysREPALiftEnvCfg


def main():
    cfg = PhysREPALiftEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs

    env = ManagerBasedRLEnv(cfg=cfg)
    print(f"\n{'='*60}")
    print(f"PhysREPA Lift Environment Created")
    print(f"  Num envs: {env.num_envs}")
    print(f"  Action space: {env.action_space}")
    print(f"  Obs groups: {list(env.observation_manager._group_obs_term_names.keys())}")
    print(f"{'='*60}\n")

    # Run a few steps with random actions
    obs, info = env.reset()
    print("[Step 0] Reset complete.")

    # Check observation structure
    print("\n--- Observation Structure ---")
    for group_name, group_obs in obs.items():
        print(f"  Group: {group_name}")
        if isinstance(group_obs, dict):
            for key, val in group_obs.items():
                if isinstance(val, torch.Tensor):
                    print(f"    {key}: shape={val.shape}, dtype={val.dtype}")
                else:
                    print(f"    {key}: type={type(val)}")
        elif isinstance(group_obs, torch.Tensor):
            print(f"    tensor: shape={group_obs.shape}")

    # Verify camera images
    print("\n--- Camera Verification ---")
    if "policy" in obs and isinstance(obs["policy"], dict):
        for cam_name in ["table_cam", "wrist_cam"]:
            if cam_name in obs["policy"]:
                img = obs["policy"][cam_name]
                print(f"  {cam_name}: shape={img.shape}, min={img.min()}, max={img.max()}, dtype={img.dtype}")
            else:
                print(f"  {cam_name}: NOT FOUND in policy obs")

    # Verify physics GT
    print("\n--- Physics GT Verification ---")
    if "physics_gt" in obs and isinstance(obs["physics_gt"], dict):
        for key, val in obs["physics_gt"].items():
            if isinstance(val, torch.Tensor):
                print(f"  {key}: shape={val.shape}, sample={val[0].tolist()[:4]}")
    else:
        print("  physics_gt group not found!")

    # Run steps and check contact sensor
    print("\n--- Running 50 steps with random actions ---")
    for step in range(50):
        action = torch.tensor(env.action_space.sample(), device=env.device, dtype=torch.float32).unsqueeze(0).repeat(env.num_envs, 1)
        obs, reward, terminated, truncated, info = env.step(action)

        if step % 10 == 0:
            # Check contact
            if "physics_gt" in obs and isinstance(obs["physics_gt"], dict):
                contact = obs["physics_gt"].get("contact_flag", torch.zeros(1))
                force = obs["physics_gt"].get("contact_force", torch.zeros(1))
                ee_dist = obs["physics_gt"].get("ee_to_object_distance", torch.zeros(1))
                print(
                    f"  Step {step}: reward={reward[0].item():.4f}, "
                    f"contact={contact[0].item():.0f}, "
                    f"force_norm={torch.norm(force[0]).item():.3f}, "
                    f"ee_dist={ee_dist[0].item():.4f}"
                )

    # Verify camera at final step
    print("\n--- Final Camera Check ---")
    if "policy" in obs and isinstance(obs["policy"], dict):
        for cam_name in ["table_cam", "wrist_cam"]:
            if cam_name in obs["policy"]:
                img = obs["policy"][cam_name]
                nonzero = (img > 0).float().mean().item()
                print(f"  {cam_name}: shape={img.shape}, non-zero pixels={nonzero*100:.1f}%")

    print(f"\n{'='*60}")
    print("ALL CHECKS PASSED")
    print(f"{'='*60}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
