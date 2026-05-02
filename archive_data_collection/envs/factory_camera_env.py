"""Factory environments with cameras for PhysREPA data collection.

Subclasses the Factory Direct env to add:
- table_cam and wrist_cam sensors
- Contact force reading (held-fixed asset contact)
- Per-episode physics randomization (friction, mass) for PEZ probing
"""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.sensors import ContactSensor, ContactSensorCfg, TiledCamera, TiledCameraCfg
from isaaclab.utils import configclass

from isaaclab_tasks.direct.factory.factory_env import FactoryEnv
from isaaclab_tasks.direct.factory.factory_env_cfg import (
    FactoryTaskPegInsertCfg,
    FactoryTaskNutThreadCfg,
)
from isaaclab_tasks.direct.factory import factory_utils


##
# Configs: inherit from Factory task configs and add cameras
##


@configclass
class PegInsertCameraCfg(FactoryTaskPegInsertCfg):
    """PegInsert with cameras. Inherits ALL Factory PegInsert settings."""

    table_cam: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/table_cam",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(1.0, 0.0, 0.20),
            rot=(-0.41345, 0.57363, 0.57363, -0.41345),
            convention="ros",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=22.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 4.0)
        ),
        width=384,
        height=384,
    )

    wrist_cam: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/panda_hand/wrist_cam",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.13, 0.0, -0.15),
            rot=(-0.70614, 0.03701, 0.03701, -0.70614),
            convention="ros",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 2.0)
        ),
        width=384,
        height=384,
    )

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/panda_leftfinger",
        update_period=0.0,
        history_length=1,
        track_air_time=True,
        track_contact_points=True,
        max_contact_data_count_per_prim=64,
        force_threshold=0.5,
        filter_prim_paths_expr=["/World/envs/env_.*/HeldAsset/forge_round_peg_8mm"],
    )

    contact_sensor_r: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/panda_rightfinger",
        update_period=0.0,
        history_length=1,
        track_air_time=False,
        track_contact_points=True,
        max_contact_data_count_per_prim=64,
        force_threshold=0.5,
        filter_prim_paths_expr=["/World/envs/env_.*/HeldAsset/forge_round_peg_8mm"],
    )

    held_fixed_contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/HeldAsset/forge_round_peg_8mm",
        update_period=0.0,
        history_length=1,
        track_air_time=False,
        track_contact_points=False,
        max_contact_data_count_per_prim=2048,
        force_threshold=0.5,
        filter_prim_paths_expr=[],
    )

    # Physics randomization ranges for PEZ probing
    held_friction_range: tuple[float, float] = (0.1, 1.2)
    fixed_friction_range: tuple[float, float] = (0.1, 1.2)
    held_mass_scale_range: tuple[float, float] = (0.5, 2.0)


@configclass
class NutThreadCameraCfg(FactoryTaskNutThreadCfg):
    """NutThread with cameras. Inherits ALL Factory NutThread settings."""

    table_cam: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/table_cam",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(1.0, 0.0, 0.20),
            rot=(-0.41345, 0.57363, 0.57363, -0.41345),
            convention="ros",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=22.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 4.0)
        ),
        width=384,
        height=384,
    )

    wrist_cam: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/panda_hand/wrist_cam",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.13, 0.0, -0.15),
            rot=(-0.70614, 0.03701, 0.03701, -0.70614),
            convention="ros",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 2.0)
        ),
        width=384,
        height=384,
    )

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/panda_leftfinger",
        update_period=0.0,
        history_length=1,
        track_air_time=True,
        track_contact_points=True,
        max_contact_data_count_per_prim=64,
        force_threshold=0.5,
        filter_prim_paths_expr=["/World/envs/env_.*/HeldAsset/factory_nut_loose"],
    )

    contact_sensor_r: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/panda_rightfinger",
        update_period=0.0,
        history_length=1,
        track_air_time=False,
        track_contact_points=True,
        max_contact_data_count_per_prim=64,
        force_threshold=0.5,
        filter_prim_paths_expr=["/World/envs/env_.*/HeldAsset/factory_nut_loose"],
    )

    held_fixed_contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/HeldAsset/factory_nut_loose",
        update_period=0.0,
        history_length=1,
        track_air_time=False,
        track_contact_points=True,
        max_contact_data_count_per_prim=2048,
        force_threshold=0.5,
        filter_prim_paths_expr=["/World/envs/env_.*/FixedAsset/factory_bolt_loose"],
    )

    # Physics randomization ranges for PEZ probing
    held_friction_range: tuple[float, float] = (0.05, 0.8)
    fixed_friction_range: tuple[float, float] = (0.05, 0.8)
    held_mass_scale_range: tuple[float, float] = (0.5, 2.0)


##
# Environment: Factory + cameras + contact force + physics randomization
##


class FactoryCameraEnv(FactoryEnv):
    """Factory env with added camera sensors, contact force reading, and physics randomization.

    Adds:
    - table_cam and wrist_cam
    - get_net_contact_forces() on held asset for peg-socket / nut-bolt contact
    - Per-episode friction and mass randomization for PEZ probing
    """

    cfg: PegInsertCameraCfg | NutThreadCameraCfg

    def _setup_scene(self):
        """Setup Factory scene then add cameras."""
        super()._setup_scene()

        # Add cameras after Factory scene is set up
        self._table_cam = TiledCamera(self.cfg.table_cam)
        self._wrist_cam = TiledCamera(self.cfg.wrist_cam)
        self.scene.sensors["table_cam"] = self._table_cam
        self.scene.sensors["wrist_cam"] = self._wrist_cam
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self._contact_sensor_r = ContactSensor(self.cfg.contact_sensor_r)
        self._held_fixed_contact_sensor = ContactSensor(self.cfg.held_fixed_contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        self.scene.sensors["contact_sensor_r"] = self._contact_sensor_r
        self.scene.sensors["held_fixed_contact_sensor"] = self._held_fixed_contact_sensor

        # Default masses stored lazily (root_physx_view not ready at scene setup time)
        self._held_default_masses = None

    def randomize_physics(self):
        """Randomize friction and mass per episode for PEZ probing.

        Call this after each reset to vary physics parameters.
        """
        # Randomize held asset friction (peg/nut)
        lo, hi = self.cfg.held_friction_range
        held_friction = lo + (hi - lo) * torch.rand(1).item()
        factory_utils.set_friction(self._held_asset, held_friction, self.num_envs)

        # Randomize fixed asset friction (hole/bolt)
        lo, hi = self.cfg.fixed_friction_range
        fixed_friction = lo + (hi - lo) * torch.rand(1).item()
        factory_utils.set_friction(self._fixed_asset, fixed_friction, self.num_envs)

        # Randomize held asset mass (scale from default)
        if self._held_default_masses is None:
            self._held_default_masses = self._held_asset.root_physx_view.get_masses().clone()
        lo, hi = self.cfg.held_mass_scale_range
        mass_scale = lo + (hi - lo) * torch.rand(1).item()
        new_masses = self._held_default_masses * mass_scale
        env_ids = torch.arange(self.num_envs, device="cpu")
        self._held_asset.root_physx_view.set_masses(new_masses, env_ids)

        return {
            "held_friction": held_friction,
            "fixed_friction": fixed_friction,
            "held_mass_scale": mass_scale,
            "held_mass": (self._held_default_masses[0, 0].item() * mass_scale),
        }

    def get_camera_data(self):
        """Get RGB images from both cameras. Returns (N, H, W, 3) uint8 tensors."""
        table_rgb = self._table_cam.data.output["rgb"][..., :3]
        wrist_rgb = self._wrist_cam.data.output["rgb"][..., :3]
        return table_rgb, wrist_rgb

    def get_physics_data(self):
        """Get physics ground truth for data collection, including contact forces.
        """
        # --- Held asset velocity (actual, not zeros) ---
        held_vel = self._held_asset.data.root_lin_vel_w
        held_angvel = self._held_asset.data.root_ang_vel_w

        def _sensor_force_and_point(sensor: ContactSensor):
            data = sensor.data
            if data.force_matrix_w is not None:
                forces = data.force_matrix_w[:, 0].sum(dim=1)
            else:
                forces = data.net_forces_w[:, 0]

            if data.contact_pos_w is not None and data.contact_pos_w.numel() > 0:
                points = torch.nan_to_num(data.contact_pos_w[:, 0, 0, :], nan=0.0)
            else:
                points = torch.zeros((self.num_envs, 3), device=self.device)
            return forces, points

        left_forces, left_points = _sensor_force_and_point(self._contact_sensor)
        right_forces, right_points = _sensor_force_and_point(self._contact_sensor_r)
        held_sensor_forces, held_sensor_points = _sensor_force_and_point(self._held_fixed_contact_sensor)

        if isinstance(self.cfg, PegInsertCameraCfg):
            # GPU pair filtering on the hole collider is unsupported in the direct Factory env.
            # Recover peg↔socket contact by removing finger-on-peg reactions from the peg body's total contact force.
            held_fixed_forces = held_sensor_forces + left_forces + right_forces
            held_fixed_points = held_sensor_points
        else:
            held_fixed_forces = held_sensor_forces
            held_fixed_points = held_sensor_points

        left_flags = (torch.norm(left_forces, dim=-1, keepdim=True) > 0.5).float()
        right_flags = (torch.norm(right_forces, dim=-1, keepdim=True) > 0.5).float()
        held_fixed_flags = (torch.norm(held_fixed_forces, dim=-1, keepdim=True) > 0.5).float()

        any_flags = ((left_flags + right_flags + held_fixed_flags) > 0).float()
        any_forces = left_forces + right_forces + held_fixed_forces
        any_points = left_points.clone()
        use_right = (left_flags.squeeze(-1) < 0.5) & (right_flags.squeeze(-1) > 0.5)
        use_held_fixed = (left_flags.squeeze(-1) < 0.5) & (right_flags.squeeze(-1) < 0.5) & (held_fixed_flags.squeeze(-1) > 0.5)
        any_points[use_right] = right_points[use_right]
        any_points[use_held_fixed] = held_fixed_points[use_held_fixed]

        return {
            "ee_position": self.fingertip_midpoint_pos.clone(),
            "ee_orientation": self.fingertip_midpoint_quat.clone(),
            "ee_velocity": self.ee_linvel_fd.clone(),
            "ee_angular_velocity": self.ee_angvel_fd.clone(),
            "held_position": self.held_pos.clone(),
            "held_orientation": self.held_quat.clone(),
            "held_velocity": held_vel.clone(),
            "held_angular_velocity": held_angvel.clone(),
            "fixed_position": self.fixed_pos.clone(),
            "fixed_orientation": self.fixed_quat.clone(),
            "joint_pos": self.joint_pos[:, :7].clone(),
            "joint_vel": self.joint_vel[:, :7].clone(),
            "contact_flag": any_flags,
            "contact_force": any_forces,
            "contact_point": any_points,
            "finger_l_contact_flag": left_flags,
            "finger_l_contact_force": left_forces,
            "finger_l_contact_point": left_points,
            "finger_r_contact_flag": right_flags,
            "finger_r_contact_force": right_forces,
            "finger_r_contact_point": right_points,
            "held_fixed_contact_flag": held_fixed_flags,
            "held_fixed_contact_force": held_fixed_forces,
            "held_fixed_contact_point": held_fixed_points,
        }
