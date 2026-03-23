"""Factory environments with cameras for PhysREPA data collection.

Subclasses the Factory Direct env to add:
- table_cam and wrist_cam sensors
- Contact force reading (held-fixed asset contact)
- Per-episode physics randomization (friction, mass) for PEZ probing
"""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCamera, TiledCameraCfg
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

        # Store default masses for scaling
        self._held_default_masses = self._held_asset.root_physx_view.get_masses().clone()

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
        """Get physics ground truth for data collection, including pair-specific contact forces."""
        # --- Held asset (peg/nut) net contact force ---
        held_contact = self._held_asset.root_physx_view.get_net_contact_forces(
            dt=self.physics_dt
        )
        held_contact_total = held_contact.sum(dim=1)  # (num_envs, 3)
        contact_mag = torch.norm(held_contact_total, dim=-1)
        contact_flag = (contact_mag > 0.5).float().unsqueeze(-1)

        # --- Finger contact forces (left/right) ---
        left_idx = self._robot.body_names.index("panda_leftfinger")
        right_idx = self._robot.body_names.index("panda_rightfinger")
        robot_contact = self._robot.root_physx_view.get_net_contact_forces(
            dt=self.physics_dt
        )
        finger_l_force = robot_contact[:, left_idx, :]   # (num_envs, 3)
        finger_r_force = robot_contact[:, right_idx, :]   # (num_envs, 3)
        finger_l_flag = (torch.norm(finger_l_force, dim=-1) > 0.5).float().unsqueeze(-1)
        finger_r_flag = (torch.norm(finger_r_force, dim=-1) > 0.5).float().unsqueeze(-1)

        # --- Fixed asset (hole/bolt) net contact force = held-fixed interaction ---
        fixed_contact = self._fixed_asset.root_physx_view.get_net_contact_forces(
            dt=self.physics_dt
        )
        fixed_contact_total = fixed_contact.sum(dim=1)  # (num_envs, 3)
        fixed_flag = (torch.norm(fixed_contact_total, dim=-1) > 0.5).float().unsqueeze(-1)

        # --- Held asset velocity (actual, not zeros) ---
        held_vel = self._held_asset.data.root_lin_vel_w
        held_angvel = self._held_asset.data.root_ang_vel_w

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
            # Legacy total contact
            "contact_flag": contact_flag,
            "contact_force": held_contact_total,
            # Pair-specific: finger L/R ↔ held asset
            "finger_l_contact_flag": finger_l_flag,
            "finger_l_contact_force": finger_l_force,
            "finger_r_contact_flag": finger_r_flag,
            "finger_r_contact_force": finger_r_force,
            # Pair-specific: held ↔ fixed (peg↔socket, nut↔bolt)
            "held_fixed_contact_flag": fixed_flag,
            "held_fixed_contact_force": fixed_contact_total,
        }
