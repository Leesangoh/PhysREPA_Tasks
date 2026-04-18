"""Quick test for PhysREPA environments."""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
from isaaclab.envs import ManagerBasedRLEnv

TASK_CONFIGS = {}


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
            if hasattr(v, 'shape'):
                print(f"  {k}: {v.shape}")
            else:
                print(f"  {k}: {type(v)}")
    else:
        print(f"  concatenated: {obs['policy'].shape}")

    print("\nPhysics GT observations:")
    if isinstance(obs["physics_gt"], dict):
        for k, v in obs["physics_gt"].items():
            if hasattr(v, 'shape'):
                print(f"  {k}: shape={v.shape}, values={v[0].cpu().numpy()[:3]}...")
            else:
                print(f"  {k}: {type(v)}")

    # Determine action dim
    if args.task == "reach":
        action_dim = 6  # IK relative only, no gripper
    else:
        action_dim = 7  # IK relative + gripper

    # Run 10 steps
    print(f"\nRunning 10 steps (action_dim={action_dim})...")
    for i in range(10):
        action = torch.zeros(1, action_dim, device=env.device)
        obs, rew, term, trunc, info = env.step(action)
        if i % 5 == 0:
            print(f"  Step {i}: reward={rew[0].item():.4f}")

    # Check cameras
    policy_obs = obs["policy"]
    if isinstance(policy_obs, dict):
        if "table_cam" in policy_obs:
            img = policy_obs["table_cam"][0].cpu().numpy()
            print(f"\ntable_cam: shape={img.shape}, dtype={img.dtype}, range=[{img.min()}, {img.max()}]")
        if "wrist_cam" in policy_obs:
            img = policy_obs["wrist_cam"][0].cpu().numpy()
            print(f"wrist_cam: shape={img.shape}, dtype={img.dtype}, range=[{img.min()}, {img.max()}]")

    # Save sample images
    import imageio
    out_dir = f"/mnt/md1/solee/data/isaac_physrepa_v2/env_test/{args.task}"
    os.makedirs(out_dir, exist_ok=True)

    if isinstance(policy_obs, dict) and "table_cam" in policy_obs:
        img = policy_obs["table_cam"][0].cpu().numpy()
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        imageio.imwrite(os.path.join(out_dir, "table_cam_sample.png"), img)
        print(f"Saved table_cam sample: {out_dir}/table_cam_sample.png")

    if isinstance(policy_obs, dict) and "wrist_cam" in policy_obs:
        img = policy_obs["wrist_cam"][0].cpu().numpy()
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        imageio.imwrite(os.path.join(out_dir, "wrist_cam_sample.png"), img)
        print(f"Saved wrist_cam sample: {out_dir}/wrist_cam_sample.png")

    print(f"\n{'='*60}")
    print(f"Test PASSED for '{args.task}'!")
    print(f"{'='*60}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
