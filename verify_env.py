"""Quick verification script for PhysREPA environments.

Loads the environment, runs a few steps, saves sample images.
"""
from __future__ import annotations

import argparse
import functools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print = functools.partial(print, flush=True)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import numpy as np
import imageio
from isaaclab.envs import ManagerBasedRLEnv


def load_task(task_name):
    if task_name == "drawer":
        from physrepa_tasks.envs.drawer_env_cfg import PhysREPADrawerEnvCfg
        return PhysREPADrawerEnvCfg
    elif task_name == "reach":
        from physrepa_tasks.envs.reach_env_cfg import PhysREPAReachEnvCfg
        return PhysREPAReachEnvCfg
    elif task_name == "push":
        from physrepa_tasks.envs.push_env_cfg import PhysREPAPushEnvCfg
        return PhysREPAPushEnvCfg
    elif task_name == "strike":
        from physrepa_tasks.envs.strike_env_cfg import PhysREPAStrikeEnvCfg
        return PhysREPAStrikeEnvCfg
    elif task_name == "lift":
        from physrepa_tasks.envs.lift_env_cfg import PhysREPALiftEnvCfg
        return PhysREPALiftEnvCfg
    elif task_name == "peg_insert":
        from physrepa_tasks.envs.peg_insert_env_cfg import PhysREPAPegInsertEnvCfg
        return PhysREPAPegInsertEnvCfg
    elif task_name == "nut_thread":
        from physrepa_tasks.envs.nut_thread_env_cfg import PhysREPANutThreadEnvCfg
        return PhysREPANutThreadEnvCfg
    else:
        raise ValueError(f"Unknown task: {task_name}")


def main():
    cfg_cls = load_task(args.task)
    cfg = cfg_cls()
    cfg.scene.num_envs = 1

    env = ManagerBasedRLEnv(cfg=cfg)
    print(f"\n{'='*60}")
    print(f"Environment '{args.task}' loaded successfully!")
    print(f"{'='*60}")

    obs, info = env.reset()

    # Print observation structure
    print("\nPolicy observations:")
    if isinstance(obs["policy"], dict):
        for k, v in obs["policy"].items():
            if hasattr(v, "shape"):
                print(f"  {k}: {v.shape}")
    else:
        print(f"  concatenated: {obs['policy'].shape}")

    if "physics_gt" in obs:
        print("\nPhysics GT observations:")
        if isinstance(obs["physics_gt"], dict):
            for k, v in obs["physics_gt"].items():
                if hasattr(v, "shape"):
                    val_str = str(v[0].cpu().numpy()[:4]) if v.numel() > 0 else "empty"
                    print(f"  {k}: shape={v.shape}, val={val_str}")

    # Determine action dim
    if args.task == "reach":
        action_dim = 6
    else:
        action_dim = 7

    # Run 20 steps with zero action
    print(f"\nRunning 20 steps (action_dim={action_dim})...")
    for i in range(20):
        action = torch.zeros(1, action_dim, device=env.device)
        obs, rew, term, trunc, info = env.step(action)
        if i % 10 == 0:
            print(f"  Step {i}: reward={rew[0].item():.4f}")

    # Save sample images
    out_dir = f"/mnt/md1/solee/data/isaac_physrepa_v2/env_test/{args.task}"
    os.makedirs(out_dir, exist_ok=True)

    policy_obs = obs["policy"]
    if isinstance(policy_obs, dict):
        for cam_name in ["table_cam", "wrist_cam"]:
            if cam_name in policy_obs:
                img = policy_obs[cam_name][0].cpu().numpy()
                if img.dtype != np.uint8:
                    img = img.astype(np.uint8)
                path = os.path.join(out_dir, f"{cam_name}_sample.png")
                imageio.imwrite(path, img)
                print(f"Saved {cam_name}: {path} (shape={img.shape}, range=[{img.min()},{img.max()}])")

    print(f"\nTest PASSED for '{args.task}'!")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
