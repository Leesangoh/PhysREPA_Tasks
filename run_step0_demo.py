"""Step 0 demo: push with 3 different friction levels, no target marker."""
from __future__ import annotations
import argparse, functools, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print = functools.partial(print, flush=True)

from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="push", choices=["push", "strike"])
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True; args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch, numpy as np, imageio
from isaaclab.envs import ManagerBasedRLEnv
from physrepa_tasks.envs.push_env_cfg import PhysREPAPushEnvCfg
from physrepa_tasks.envs.strike_env_cfg import PhysREPAStrikeEnvCfg
from physrepa_tasks.policies.scripted_policy import Step0PushPolicy, Step0StrikePolicy

FRICTION_CONFIGS = [
    {"label": "low_friction", "surface": 0.1, "object": 0.2},
    {"label": "mid_friction", "surface": 0.5, "object": 0.5},
    {"label": "high_friction", "surface": 0.9, "object": 0.8},
]

def encode_video(frames, path, fps):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    w = imageio.get_writer(path, fps=fps, codec="libx264", quality=8, pixelformat="yuv420p")
    for f in frames: w.append_data(f)
    w.close()

def main():
    if args.task == "push":
        cfg = PhysREPAPushEnvCfg()
        PolicyCls = Step0PushPolicy
    else:
        cfg = PhysREPAStrikeEnvCfg()
        PolicyCls = Step0StrikePolicy

    cfg.scene.num_envs = 1
    # Remove target marker by moving underground
    if hasattr(cfg.scene, 'target_marker'):
        cfg.scene.target_marker.init_state.pos = [0, 0, -10]

    env = ManagerBasedRLEnv(cfg=cfg)
    dt = cfg.sim.dt * cfg.decimation
    fps = int(1.0 / dt)
    max_steps = int(cfg.episode_length_s / dt)

    policy = PolicyCls(num_envs=1, device=env.device)
    out_dir = f"/mnt/md1/solee/data/isaac_physrepa_v2/step0_demo/{args.task}"
    os.makedirs(out_dir, exist_ok=True)

    for i, fc in enumerate(FRICTION_CONFIGS):
        print(f"\n=== Episode {i}: {fc['label']} (surface={fc['surface']}, object={fc['object']}) ===")
        obs, _ = env.reset()

        # Override friction
        obj = env.scene["object"]
        mat = obj.root_physx_view.get_material_properties().clone()
        mat[:, :, 0] = fc["object"]  # static
        mat[:, :, 1] = fc["object"]  # dynamic
        indices = torch.tensor([0], dtype=torch.int32, device="cpu")
        obj.root_physx_view.set_material_properties(mat, indices)

        surface = env.scene["surface"]
        smat = surface.root_physx_view.get_material_properties().clone()
        smat[:, :, 0] = fc["surface"]
        smat[:, :, 1] = fc["surface"]
        surface.root_physx_view.set_material_properties(smat, indices)

        policy.reset()

        # Warmup
        warmup = int(0.5 / dt)
        for _ in range(warmup):
            obs, _, _, _, _ = env.step(torch.zeros(1, 7, device=env.device))
        policy.reset()

        frames = []
        for step in range(max_steps - warmup):
            action = policy.get_action(obs)
            obs, rew, _, _, _ = env.step(action)
            img = obs["policy"]["table_cam"][0].cpu().numpy()
            if img.dtype != np.uint8: img = img.astype(np.uint8)
            frames.append(img)
            if step % 100 == 0:
                obj_pos = obs["physics_gt"]["object_position"][0].cpu().numpy()
                print(f"  Step {step}: obj=({obj_pos[0]:.3f},{obj_pos[1]:.3f})")

        path = os.path.join(out_dir, f"{fc['label']}.mp4")
        encode_video(frames, path, fps)
        print(f"  Saved: {path} ({os.path.getsize(path)/1024:.1f} KB)")

        # Save key frames
        for fi, label in [(0, "start"), (len(frames)//3, "push"), (len(frames)-1, "end")]:
            imageio.imwrite(os.path.join(out_dir, f"{fc['label']}_{label}.png"), frames[fi])

    print(f"\nDone. Output: {out_dir}")
    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
