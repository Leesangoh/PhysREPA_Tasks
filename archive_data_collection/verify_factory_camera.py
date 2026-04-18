"""Verify Factory+Camera env works correctly."""
from __future__ import annotations

import argparse
import functools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print = functools.partial(print, flush=True)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True, choices=["peg_insert", "nut_thread"])
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import numpy as np
import imageio

from physrepa_tasks.envs.factory_camera_env import (
    FactoryCameraEnv,
    PegInsertCameraCfg,
    NutThreadCameraCfg,
)


def main():
    if args.task == "peg_insert":
        cfg = PegInsertCameraCfg()
    else:
        cfg = NutThreadCameraCfg()

    cfg.scene.num_envs = 1

    env = FactoryCameraEnv(cfg=cfg)
    print(f"\n{'='*60}")
    print(f"Factory+Camera env '{args.task}' loaded!")
    print(f"{'='*60}")

    obs, info = env.reset()
    print(f"Obs shape: {obs['policy'].shape}")

    # Run 20 steps
    for i in range(20):
        action = torch.zeros(1, 6, device=env.device)
        obs, rew, term, trunc, info = env.step(action)

    # Get camera images
    table_rgb, wrist_rgb = env.get_camera_data()
    print(f"table_cam: {table_rgb.shape}, wrist_cam: {wrist_rgb.shape}")

    # Get physics data
    phys = env.get_physics_data()
    print(f"EE pos: {phys['ee_position'][0].cpu().numpy()}")
    print(f"Held pos: {phys['held_position'][0].cpu().numpy()}")
    print(f"Fixed pos: {phys['fixed_position'][0].cpu().numpy()}")
    ee_held_dist = torch.norm(phys['ee_position'][0] - phys['held_position'][0]).item()
    print(f"EE-Held distance: {ee_held_dist:.4f}m (should be ~0.03m if grasped)")

    # Save images
    out_dir = f"/mnt/md1/solee/data/isaac_physrepa_v2/env_test/{args.task}"
    os.makedirs(out_dir, exist_ok=True)

    table_img = table_rgb[0].cpu().numpy().astype(np.uint8)
    wrist_img = wrist_rgb[0].cpu().numpy().astype(np.uint8)
    imageio.imwrite(os.path.join(out_dir, "factory_table_cam.png"), table_img)
    imageio.imwrite(os.path.join(out_dir, "factory_wrist_cam.png"), wrist_img)
    print(f"Saved to {out_dir}")

    print(f"\nTest PASSED for '{args.task}'!")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
