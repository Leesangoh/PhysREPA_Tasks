"""Event function to sync target marker position with command target."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def sync_target_marker(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    command_name: str = "object_pose",
    marker_name: str = "target_marker",
    fixed_z: float | None = 0.005,
):
    """Move the visual target marker to match the command target position.

    Only modifies the existing xformOp:translate value — does NOT add/remove ops.

    Args:
        fixed_z: If not None, override the z coordinate (for table-surface markers).
                 If None, use the command's z coordinate (for 3D targets like reach).
    """
    import omni.usd
    from pxr import Gf

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)

    # Get command target position (in robot base frame)
    command = env.command_manager.get_command(command_name)
    target_pos_b = command[:, :3]

    # Convert to world frame
    from isaaclab.utils.math import combine_frame_transforms
    robot = env.scene["robot"]
    target_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, target_pos_b
    )

    stage = omni.usd.get_context().get_stage()

    for env_id in env_ids.cpu().tolist():
        pos = target_pos_w[env_id].cpu().tolist()
        prim_path = f"/World/envs/env_{env_id}/TargetMarker"
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            continue

        z = fixed_z if fixed_z is not None else pos[2]
        # Only SET the existing translate attribute — don't modify xformOpOrder
        attr = prim.GetAttribute("xformOp:translate")
        if attr.IsValid():
            attr.Set(Gf.Vec3d(pos[0], pos[1], z))
