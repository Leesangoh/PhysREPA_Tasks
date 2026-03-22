"""Factory environments with cameras for PhysREPA data collection.

Subclasses the Factory Direct env to add table_cam and wrist_cam sensors
while keeping all Factory grasp/physics mechanics IDENTICAL.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.utils import configclass

from isaaclab_tasks.direct.factory.factory_env import FactoryEnv
from isaaclab_tasks.direct.factory.factory_env_cfg import (
    FactoryTaskPegInsertCfg,
    FactoryTaskNutThreadCfg,
)


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


##
# Environment: Factory + cameras
##


class FactoryCameraEnv(FactoryEnv):
    """Factory env with added camera sensors.

    Identical to FactoryEnv in every way — just adds table_cam and wrist_cam.
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

    def get_camera_data(self):
        """Get RGB images from both cameras. Returns (N, H, W, 3) uint8 tensors."""
        table_rgb = self._table_cam.data.output["rgb"][..., :3]
        wrist_rgb = self._wrist_cam.data.output["rgb"][..., :3]
        return table_rgb, wrist_rgb

    def get_physics_data(self):
        """Get physics ground truth for data collection."""
        return {
            "ee_position": self.fingertip_midpoint_pos.clone(),
            "ee_velocity": self.ee_linvel_fd.clone(),
            "held_position": self.held_pos.clone(),
            "held_orientation": self.held_quat.clone(),
            "fixed_position": self.fixed_pos.clone(),
            "fixed_orientation": self.fixed_quat.clone(),
            "joint_pos": self.joint_pos[:, :7].clone(),
            "joint_vel": self.joint_vel[:, :7].clone(),
        }
