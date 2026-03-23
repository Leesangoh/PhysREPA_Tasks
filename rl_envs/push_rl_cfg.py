# Copyright (c) 2024, PhysREPA Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Push RL environment configuration for Franka with joint position control.

Modifications from Pick-Place:
- Gripper forced open (both open and close commands set to 0.04)
- Target on table surface (z=0.02)
- No lifting reward (minimal_height=0.0)
- Push-specific rewards: approach, push progress, object at target
"""

from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.sim.schemas.schemas_cfg import MassPropertiesCfg, RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.sim.spawners.shapes.shapes_cfg import CuboidCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import combine_frame_transforms

import torch
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# MDP imports — reuse existing lift MDP functions
from isaaclab_tasks.manager_based.manipulation.lift import mdp
# Import custom observation functions (friction, mass)
from physrepa_tasks.mdp.observations import object_friction_obs, surface_friction_obs, object_mass_obs
mdp.object_friction_obs = object_friction_obs
mdp.surface_friction_obs = surface_friction_obs
mdp.object_mass_obs = object_mass_obs

# Pre-defined configs
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip

##
# Scene definition
##


@configclass
class PushSceneCfg(InteractiveSceneCfg):
    """Scene configuration for pushing with Franka and a cuboid object."""

    # Robot
    robot: ArticulationCfg = MISSING
    # End-effector sensor
    ee_frame: FrameTransformerCfg = MISSING
    # Target object
    object: RigidObjectCfg = MISSING

    # Thin surface on table for friction randomization
    surface: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Surface",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.001], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(
            size=(0.6, 0.8, 0.002),
            rigid_props=RigidBodyPropertiesCfg(
                disable_gravity=True,
                kinematic_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    # Ground plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    # Target zone — blue disc, r=8cm
    target_marker = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TargetMarker",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0.0, 0.005]),
        spawn=sim_utils.CylinderCfg(
            radius=0.08,
            height=0.005,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.3, 0.9)),
        ),
    )

    # Lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for pushing: target position on the table surface."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="panda_hand",
        resampling_time_range=(12.0, 12.0),
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.35, 0.65),
            pos_y=(-0.20, 0.20),
            pos_z=(0.02, 0.02),  # On the table surface
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications: IK relative control for arm + gripper closed."""

    arm_action: DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for pushing."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)
        # Physics-aware observations for adaptive pushing
        object_friction = ObsTerm(func=mdp.object_friction_obs)
        surface_friction = ObsTerm(func=mdp.surface_friction_obs)
        object_mass = ObsTerm(func=mdp.object_mass_obs)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events including physics randomization."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )

    # Physics randomization: mass
    randomize_object_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (0.05, 2.0),
            "operation": "abs",
        },
    )

    # Physics randomization: friction
    randomize_object_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": (0.2, 0.8),
            "dynamic_friction_range": (0.2, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # Surface friction randomization (crucial for push dynamics)
    randomize_surface_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("surface"),
            "static_friction_range": (0.1, 0.8),
            "dynamic_friction_range": (0.1, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
            "make_consistent": True,
        },
    )


##
# Custom push reward functions
##


def _get_target_pos_w(env, command_name: str = "object_pose"):
    """Helper: get target position in world frame."""
    robot = env.scene["robot"]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    return des_pos_w


def reaching_behind_object(
    env, std: float, offset: float = 0.08, command_name: str = "object_pose",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward EE for reaching the push start position (behind the object, opposite from target)."""
    obj = env.scene[object_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    target_w = _get_target_pos_w(env, command_name)

    obj_pos = obj.data.root_pos_w[:, :2]
    ee_pos = ee_frame.data.target_pos_w[..., 0, :2]

    # Push direction: object → target (XY)
    push_dir = target_w[:, :2] - obj_pos
    push_dir = push_dir / push_dir.norm(dim=1, keepdim=True).clamp(min=1e-6)

    # Behind position: object - push_dir * offset
    behind_pos = obj_pos - push_dir * offset

    distance = torch.norm(ee_pos - behind_pos, dim=1)
    return 1 - torch.tanh(distance / std)


def ee_at_push_height(
    env, target_height: float = 0.05, std: float = 0.05,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward EE for being at the correct push height (near table surface)."""
    ee_frame = env.scene[ee_frame_cfg.name]
    ee_z = ee_frame.data.target_pos_w[..., 0, 2]
    height_error = torch.abs(ee_z - target_height)
    return 1 - torch.tanh(height_error / std)


def object_target_distance(
    env, std: float, command_name: str = "object_pose",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward for reducing object-to-target distance (XY only)."""
    obj = env.scene[object_cfg.name]
    target_w = _get_target_pos_w(env, command_name)
    distance = torch.norm(target_w[:, :2] - obj.data.root_pos_w[:, :2], dim=1)
    return 1 - torch.tanh(distance / std)


def object_height_penalty(
    env, max_height: float = 0.08,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Penalize object being lifted off table."""
    obj = env.scene[object_cfg.name]
    obj_z = obj.data.root_pos_w[:, 2]
    return -torch.clamp(obj_z - max_height, min=0.0)


def object_velocity_penalty(
    env, max_speed: float = 0.3,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Penalize object moving too fast — encourages gentle pushing instead of hitting."""
    obj = env.scene[object_cfg.name]
    speed = torch.norm(obj.data.root_lin_vel_w[:, :2], dim=1)
    return -torch.clamp(speed - max_speed, min=0.0)


def ee_object_proximity(
    env, std: float = 0.05,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward EE staying close to object — encourages sustained contact pushing."""
    obj = env.scene[object_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    ee_pos = ee_frame.data.target_pos_w[..., 0, :2]
    obj_pos = obj.data.root_pos_w[:, :2]
    distance = torch.norm(ee_pos - obj_pos, dim=1)
    return 1 - torch.tanh(distance / std)


@configclass
class RewardsCfg:
    """Reward terms for pushing — v4: prioritize actually moving the object."""

    # 1. EE reaches behind object (lowered — positioning aid only)
    reaching_behind = RewTerm(
        func=reaching_behind_object,
        params={"std": 0.1, "offset": 0.08, "command_name": "object_pose"},
        weight=5.0,
    )

    # 2. EE at correct push height
    push_height = RewTerm(
        func=ee_at_push_height,
        params={"target_height": 0.03, "std": 0.03},
        weight=5.0,
    )

    # 3. Object-target distance (coarse) — PRIMARY reward
    object_target_dist = RewTerm(
        func=object_target_distance,
        params={"std": 0.1, "command_name": "object_pose"},
        weight=25.0,
    )

    # 4. Object-target distance (fine)
    object_target_dist_fine = RewTerm(
        func=object_target_distance,
        params={"std": 0.03, "command_name": "object_pose"},
        weight=30.0,
    )

    # 5. Height penalty — don't lift object
    height_penalty = RewTerm(
        func=object_height_penalty,
        params={"max_height": 0.08},
        weight=10.0,
    )

    # 6. EE-object proximity — reward staying close to object during push
    ee_proximity = RewTerm(
        func=ee_object_proximity,
        params={"std": 0.05},
        weight=5.0,
    )

    # Action/joint penalties (small)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-5)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for pushing."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for pushing."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000}
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000}
    )


##
# Environment configuration
##


@configclass
class PushRLEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the push RL environment."""

    # Scene settings
    scene: PushSceneCfg = PushSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # General settings
        self.decimation = 2
        self.episode_length_s = 10.0
        # Simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

        # Set Franka as robot (HIGH_PD for IK tracking)
        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # IK Relative control — 6D delta EE pose
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=0.2,  # smaller scale to prevent divergence
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )
        # Gripper closed fist
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.01},
            close_command_expr={"panda_finger_.*": 0.01},
        )

        # Set cuboid object (0.06m red cube)
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.055], rot=[1, 0, 0, 0]),
            spawn=CuboidCfg(
                size=(0.06, 0.06, 0.06),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=64,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
                mass_props=MassPropertiesCfg(density=400.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1)),  # Red
            ),
        )

        # End-effector frame transformer
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.1034]),
                ),
            ],
        )


@configclass
class PushRLEnvCfg_PLAY(PushRLEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
