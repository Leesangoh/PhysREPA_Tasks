"""Custom event functions for PhysREPA tasks."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# Available colors for object randomization
COLORS = {
    "red": (0.9, 0.1, 0.1),
    "green": (0.1, 0.8, 0.1),
    "blue": (0.1, 0.2, 0.9),
    "yellow": (0.9, 0.9, 0.1),
    "orange": (1.0, 0.5, 0.0),
    "purple": (0.6, 0.1, 0.8),
    "cyan": (0.0, 0.8, 0.8),
    "white": (0.9, 0.9, 0.9),
}

COLOR_NAMES = list(COLORS.keys())


def randomize_object_visual_color(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_prim_path: str = "{ENV_REGEX_NS}/Object",
    material_path_suffix: str = "/geometry/material/Shader",
):
    """Randomize the diffuse color of a spawned object at reset.

    Works with objects spawned via CuboidCfg/CylinderCfg/SphereCfg that have
    a PreviewSurface material. Changes the diffuseColor input on the shader.

    The selected color name is stored in env._physrepa_object_colors dict
    keyed by env_id for retrieval by the data collection script.

    Args:
        env: The environment instance.
        env_ids: Environment indices to randomize.
        asset_prim_path: Prim path pattern for the object.
        material_path_suffix: Suffix to find the PreviewSurface shader.
    """
    import omni.usd
    from pxr import Sdf, Gf

    stage = omni.usd.get_context().get_stage()

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)

    # Initialize color storage if not present
    if not hasattr(env, "_physrepa_object_colors"):
        env._physrepa_object_colors = {}

    for env_id in env_ids.cpu().tolist():
        # Pick random color
        color_name = random.choice(COLOR_NAMES)
        color_rgb = COLORS[color_name]
        env._physrepa_object_colors[env_id] = color_name

        # Resolve prim path for this env
        env_prim_path = asset_prim_path.replace("{ENV_REGEX_NS}", f"/World/envs/env_{env_id}")
        shader_path = env_prim_path + material_path_suffix

        shader_prim = stage.GetPrimAtPath(shader_path)
        if shader_prim.IsValid():
            color_attr = shader_prim.GetAttribute("inputs:diffuseColor")
            if color_attr.IsValid():
                color_attr.Set(Gf.Vec3f(*color_rgb))


def randomize_rigid_body_damping(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    linear_damping_range: tuple[float, float] = (3.0, 8.0),
    angular_damping_ratio: float = 0.5,
):
    """Randomize linear/angular damping of a rigid body via USD API.

    Args:
        asset_cfg: Asset to randomize.
        linear_damping_range: Uniform range for linear_damping.
        angular_damping_ratio: angular_damping = linear_damping * this ratio.
    """
    import omni.usd
    from pxr import UsdPhysics

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)

    stage = omni.usd.get_context().get_stage()
    lo, hi = linear_damping_range

    for env_id in env_ids.cpu().tolist():
        lin_damp = random.uniform(lo, hi)
        ang_damp = lin_damp * angular_damping_ratio

        # Find the rigid body prim under the asset
        asset_path = asset_cfg.name
        env_prim_path = f"/World/envs/env_{env_id}/{asset_path.capitalize()}"
        prim = stage.GetPrimAtPath(env_prim_path)
        if not prim.IsValid():
            # Try alternative path patterns
            for child in stage.GetPrimAtPath(f"/World/envs/env_{env_id}").GetChildren():
                if asset_path.lower() in child.GetName().lower():
                    prim = child
                    break

        if not prim.IsValid():
            continue

        # Find rigid body API on prim or children
        for p in [prim] + list(prim.GetAllChildren()):
            rb_api = UsdPhysics.RigidBodyAPI(p)
            if rb_api:
                # Set damping via PhysxRigidBodyAPI
                from pxr import PhysxSchema
                physx_rb = PhysxSchema.PhysxRigidBodyAPI(p)
                if not physx_rb:
                    physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(p)
                physx_rb.GetLinearDampingAttr().Set(lin_damp)
                physx_rb.GetAngularDampingAttr().Set(ang_damp)
                break


def teleport_object_to_ee(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("peg"),
    held_asset_height: float = 0.05,
    fingerpad_length: float = 0.017608,
):
    """Place held object in gripper and close gripper to grasp.

    Replicates Factory env's pre-grasp initialization:
    1. Compute exact peg/nut position relative to fingertip using Factory's transform math
    2. Place object at computed position
    3. Step sim with closed gripper to establish physical contact

    Args:
        env: The environment instance.
        env_ids: Environment indices to reset.
        asset_cfg: The asset to teleport (e.g., "peg" or "nut").
        held_asset_height: Height of the held object (peg=0.05, nut=0.01).
        fingerpad_length: Franka fingerpad length (0.017608m).
    """
    import isaacsim.core.utils.torch as torch_utils

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)

    num_envs = env.scene.num_envs
    device = env.device
    robot = env.scene["robot"]

    # Get fingertip midpoint (same as Factory)
    lf_idx = robot.body_names.index("panda_leftfinger")
    rf_idx = robot.body_names.index("panda_rightfinger")
    fingertip_pos = (robot.data.body_pos_w[:, lf_idx] + robot.data.body_pos_w[:, rf_idx]) / 2.0
    fingertip_quat = robot.data.body_quat_w[:, lf_idx]  # approx

    # Subtract env origins to get local pos (Factory uses local coords)
    fingertip_pos_local = fingertip_pos - env.scene.env_origins

    # Flip gripper z orientation (Factory convention)
    flip_z_quat = torch.tensor([0.0, 0.0, 1.0, 0.0], device=device).unsqueeze(0).repeat(num_envs, 1)
    fingertip_flipped_quat, fingertip_flipped_pos = torch_utils.tf_combine(
        q1=fingertip_quat, t1=fingertip_pos_local,
        q2=flip_z_quat, t2=torch.zeros((num_envs, 3), device=device),
    )

    # Compute held asset relative position (from Factory's get_handheld_asset_relative_pose)
    held_rel_pos = torch.zeros((num_envs, 3), device=device)
    held_rel_pos[:, 2] = held_asset_height - fingerpad_length
    held_rel_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).unsqueeze(0).repeat(num_envs, 1)

    # Inverse transform
    asset_in_hand_quat, asset_in_hand_pos = torch_utils.tf_inverse(held_rel_quat, held_rel_pos)

    # Final world position
    translated_quat, translated_pos = torch_utils.tf_combine(
        q1=fingertip_flipped_quat, t1=fingertip_flipped_pos,
        q2=asset_in_hand_quat, t2=asset_in_hand_pos,
    )

    # Write to sim
    obj = env.scene[asset_cfg.name]
    obj_state = obj.data.default_root_state.clone()
    obj_state[env_ids, 0:3] = translated_pos[env_ids] + env.scene.env_origins[env_ids]
    obj_state[env_ids, 3:7] = translated_quat[env_ids]
    obj_state[env_ids, 7:] = 0.0
    obj.write_root_pose_to_sim(obj_state[:, 0:7], env_ids=env_ids)
    obj.write_root_velocity_to_sim(obj_state[:, 7:], env_ids=env_ids)

    # Close gripper and step sim to establish grasp (like Factory)
    gripper_close_pos = torch.zeros_like(robot.data.joint_pos)
    gripper_close_pos[:] = robot.data.joint_pos[:]
    gripper_close_pos[:, -2:] = 0.0  # close fingers fully

    grasp_steps = int(0.25 / env.sim.get_physics_dt())
    for _ in range(grasp_steps):
        robot.set_joint_position_target(gripper_close_pos)
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(dt=env.sim.get_physics_dt())
