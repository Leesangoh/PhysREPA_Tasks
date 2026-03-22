"""Custom observation functions for PhysREPA tasks."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    object_pos_w = obj.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, object_pos_w)
    return object_pos_b


def object_position_w(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Object position in world frame (3D)."""
    obj: RigidObject = env.scene[object_cfg.name]
    return obj.data.root_pos_w[:, :3]


def object_orientation_w(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Object orientation in world frame (quaternion wxyz, 4D)."""
    obj: RigidObject = env.scene[object_cfg.name]
    return obj.data.root_quat_w


def object_velocity_w(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Object linear velocity in world frame (3D)."""
    obj: RigidObject = env.scene[object_cfg.name]
    return obj.data.root_lin_vel_w


def object_angular_velocity_w(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Object angular velocity in world frame (3D)."""
    obj: RigidObject = env.scene[object_cfg.name]
    return obj.data.root_ang_vel_w


def ee_position_w(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """End-effector position in world frame (3D)."""
    ee_frame = env.scene[ee_frame_cfg.name]
    return ee_frame.data.target_pos_w[..., 0, :]


def ee_position_b(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """End-effector position in robot root frame (3D). Matches IK controller's frame."""
    ee_frame = env.scene[ee_frame_cfg.name]
    robot = env.scene[robot_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    ee_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, ee_pos_w)
    return ee_pos_b


def ee_velocity_w(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """End-effector linear velocity in world frame (3D).

    Approximated from body velocity of panda_hand.
    """
    robot = env.scene[robot_cfg.name]
    # body index for panda_hand
    hand_idx = robot.find_bodies("panda_hand")[0][0]
    return robot.data.body_lin_vel_w[:, hand_idx, :]


def ee_to_object_distance(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Euclidean distance between end-effector and object (scalar per env)."""
    ee_frame = env.scene[ee_frame_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    ee_pos = ee_frame.data.target_pos_w[..., 0, :]
    obj_pos = obj.data.root_pos_w[:, :3]
    return torch.norm(ee_pos - obj_pos, dim=-1, keepdim=True)


def contact_force(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor"),
) -> torch.Tensor:
    """Contact force from the contact sensor (3D normal force)."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return sensor.data.net_forces_w[:, 0, :]


def ee_acceleration_w(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """EE acceleration via finite difference of velocity. Returns (num_envs, 3)."""
    robot = env.scene[robot_cfg.name]
    hand_idx = robot.find_bodies("panda_hand")[0][0]
    current_vel = robot.data.body_lin_vel_w[:, hand_idx, :]

    if not hasattr(env, "_physrepa_prev_ee_vel"):
        env._physrepa_prev_ee_vel = current_vel.clone()

    dt = env.step_dt
    acceleration = (current_vel - env._physrepa_prev_ee_vel) / dt
    env._physrepa_prev_ee_vel = current_vel.clone()
    return acceleration


def object_acceleration_w(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Object acceleration via finite difference of velocity. Returns (num_envs, 3)."""
    obj: RigidObject = env.scene[object_cfg.name]
    current_vel = obj.data.root_lin_vel_w

    attr_name = f"_physrepa_prev_{object_cfg.name}_vel"
    if not hasattr(env, attr_name):
        setattr(env, attr_name, current_vel.clone())

    prev_vel = getattr(env, attr_name)
    dt = env.step_dt
    acceleration = (current_vel - prev_vel) / dt
    setattr(env, attr_name, current_vel.clone())
    return acceleration


def object_surface_contact(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    surface_height: float = 0.0,
    threshold: float = 0.005,
    half_size: float = 0.05,
) -> torch.Tensor:
    """Check if object is in contact with table surface. Returns (num_envs, 1) float."""
    obj: RigidObject = env.scene[object_cfg.name]
    obj_z = obj.data.root_pos_w[:, 2]
    on_surface = (obj_z < (surface_height + threshold + half_size)).float()
    return on_surface.unsqueeze(-1)


def object_object_distance(
    env: ManagerBasedRLEnv,
    object_a_cfg: SceneEntityCfg = SceneEntityCfg("cube_a"),
    object_b_cfg: SceneEntityCfg = SceneEntityCfg("cube_b"),
) -> torch.Tensor:
    """Distance between two objects. Returns (num_envs, 1)."""
    obj_a: RigidObject = env.scene[object_a_cfg.name]
    obj_b: RigidObject = env.scene[object_b_cfg.name]
    pos_a = obj_a.data.root_pos_w[:, :3]
    pos_b = obj_b.data.root_pos_w[:, :3]
    return torch.norm(pos_a - pos_b, dim=-1, keepdim=True)


def contact_point_w(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor"),
) -> torch.Tensor:
    """Contact point position in world frame (3D). NaN if no contact."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # Shape: (N, B, M, 3) -> take first body, first filter
    contact_pos = sensor.data.contact_pos_w[:, 0, 0, :]
    # Replace NaN with zeros for parquet storage
    contact_pos = torch.nan_to_num(contact_pos, nan=0.0)
    return contact_pos


def contact_flag(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor"),
    threshold: float = 1.0,
) -> torch.Tensor:
    """Binary contact flag (1 if contact force > threshold, else 0)."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force_magnitude = torch.norm(sensor.data.net_forces_w[:, 0, :], dim=-1, keepdim=True)
    return (force_magnitude > threshold).float()


def object_friction_obs(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Return current object friction coefficients as observation (num_envs, 2)."""
    obj: RigidObject = env.scene[object_cfg.name]
    mat = obj.root_physx_view.get_material_properties()
    return mat[:, 0, :2].to(env.device)


def surface_friction_obs(
    env: ManagerBasedRLEnv,
    surface_cfg: SceneEntityCfg = SceneEntityCfg("surface"),
) -> torch.Tensor:
    """Return current surface friction coefficients as observation (num_envs, 2)."""
    surface: RigidObject = env.scene[surface_cfg.name]
    mat = surface.root_physx_view.get_material_properties()
    return mat[:, 0, :2].to(env.device)


def object_mass_obs(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Return current object mass as observation (num_envs, 1)."""
    obj: RigidObject = env.scene[object_cfg.name]
    mass = obj.root_physx_view.get_masses()
    return mass[:, 0:1].to(env.device)
