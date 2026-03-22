# Copyright (c) 2024, PhysREPA Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Stack RL environment configuration for Franka with joint position control.

Two-cube stacking task: pick up cube_a (small, red) and stack it on cube_b (large, blue).
Uses custom reward and observation functions for multi-object tracking.
"""

from __future__ import annotations

from dataclasses import MISSING

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformer, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import MassPropertiesCfg, RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.sim.spawners.shapes.shapes_cfg import CuboidCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import subtract_frame_transforms

# MDP imports — reuse existing lift MDP functions
from isaaclab_tasks.manager_based.manipulation.lift import mdp

# Pre-defined configs
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip

##
# Custom observation and reward functions for stacking
##


def cube_a_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube_a"),
) -> torch.Tensor:
    """The position of cube_a in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    cube_a: RigidObject = env.scene[object_cfg.name]
    cube_a_pos_w = cube_a.data.root_pos_w[:, :3]
    cube_a_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, cube_a_pos_w)
    return cube_a_pos_b


def cube_b_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube_b"),
) -> torch.Tensor:
    """The position of cube_b in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    cube_b: RigidObject = env.scene[object_cfg.name]
    cube_b_pos_w = cube_b.data.root_pos_w[:, :3]
    cube_b_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, cube_b_pos_w)
    return cube_b_pos_b


def cube_a_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube_a"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching cube_a using tanh-kernel."""
    cube_a: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    cube_a_pos_w = cube_a.data.root_pos_w
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    distance = torch.norm(cube_a_pos_w - ee_w, dim=1)
    return 1 - torch.tanh(distance / std)


def cube_a_is_lifted(
    env: ManagerBasedRLEnv,
    scale: float = 0.1,
    table_height: float = 0.03,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube_a"),
) -> torch.Tensor:
    """Continuous reward proportional to cube_a height above table. Any lift is rewarded."""
    cube_a: RigidObject = env.scene[object_cfg.name]
    height_above = torch.clamp(cube_a.data.root_pos_w[:, 2] - table_height, min=0.0)
    return torch.tanh(height_above / scale)


def cube_a_grasp_reward(
    env: ManagerBasedRLEnv,
    std: float = 0.04,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube_a"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Continuous reward for grasping: proximity * gripper closure.

    Both components are continuous — gives gradient for partial grasp attempts.
    """
    cube_a: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    robot = env.scene[robot_cfg.name]

    # Continuous proximity (tanh kernel)
    ee_pos = ee_frame.data.target_pos_w[..., 0, :]
    distance = torch.norm(cube_a.data.root_pos_w - ee_pos, dim=1)
    proximity = 1 - torch.tanh(distance / std)

    # Continuous gripper closure (0=open at 0.04, 1=closed at 0.0)
    finger_ids = robot.find_joints("panda_finger.*")[0]
    finger_pos = robot.data.joint_pos[:, finger_ids].mean(dim=1)
    gripper_closure = 1.0 - torch.clamp(finger_pos / 0.04, 0.0, 1.0)

    return proximity * gripper_closure


def cube_a_above_cube_b(
    env: ManagerBasedRLEnv,
    std: float,
    min_height: float = 0.05,
    cube_a_cfg: SceneEntityCfg = SceneEntityCfg("cube_a"),
    cube_b_cfg: SceneEntityCfg = SceneEntityCfg("cube_b"),
) -> torch.Tensor:
    """Reward for cube_a being above cube_b. Soft lift gating (continuous)."""
    cube_a: RigidObject = env.scene[cube_a_cfg.name]
    cube_b: RigidObject = env.scene[cube_b_cfg.name]
    xy_distance = torch.norm(cube_a.data.root_pos_w[:, :2] - cube_b.data.root_pos_w[:, :2], dim=1)
    # Soft lift gating: continuous, not binary
    lift_amount = torch.clamp(cube_a.data.root_pos_w[:, 2] - min_height, min=0.0)
    lift_factor = torch.tanh(lift_amount / 0.05)  # smooth 0→1
    return lift_factor * (1 - torch.tanh(xy_distance / std))


def stacking_success(
    env: ManagerBasedRLEnv,
    std: float,
    stack_height_offset: float = 0.06,
    cube_a_cfg: SceneEntityCfg = SceneEntityCfg("cube_a"),
    cube_b_cfg: SceneEntityCfg = SceneEntityCfg("cube_b"),
) -> torch.Tensor:
    """Reward for successfully stacking cube_a on top of cube_b.

    Checks both XY alignment and correct Z height (cube_a should be at cube_b_z + offset).
    """
    cube_a: RigidObject = env.scene[cube_a_cfg.name]
    cube_b: RigidObject = env.scene[cube_b_cfg.name]
    # Target position for cube_a: on top of cube_b
    target_pos = cube_b.data.root_pos_w.clone()
    target_pos[:, 2] += stack_height_offset  # cube_b_height/2 + cube_a_height/2
    # 3D distance
    distance = torch.norm(cube_a.data.root_pos_w - target_pos, dim=1)
    return 1 - torch.tanh(distance / std)


def cube_a_dropping(
    env: ManagerBasedRLEnv,
    minimum_height: float = -0.05,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube_a"),
) -> torch.Tensor:
    """Termination if cube_a falls below minimum height."""
    cube_a: RigidObject = env.scene[object_cfg.name]
    return cube_a.data.root_pos_w[:, 2] < minimum_height


##
# Scene definition
##


@configclass
class StackSceneCfg(InteractiveSceneCfg):
    """Scene configuration for stacking with Franka and two cuboid objects."""

    # Robot
    robot: ArticulationCfg = MISSING
    # End-effector sensor
    ee_frame: FrameTransformerCfg = MISSING
    # Two cubes
    cube_a: RigidObjectCfg = MISSING  # Small cube to pick up (red)
    cube_b: RigidObjectCfg = MISSING  # Large base cube (blue)

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
    """Command terms for stacking: target above cube_b.

    Note: The command is defined but the actual stacking target is computed dynamically
    relative to cube_b. The command serves as a placeholder for the manager framework.
    """

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="panda_hand",
        resampling_time_range=(12.0, 12.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.4, 0.6),
            pos_y=(-0.15, 0.15),
            pos_z=(0.12, 0.12),  # Approximate stack height
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications: joint position control for arm + binary gripper."""

    arm_action: mdp.JointPositionActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for stacking."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        cube_a_position = ObsTerm(func=cube_a_position_in_robot_root_frame)
        cube_b_position = ObsTerm(func=cube_b_position_in_robot_root_frame)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events including physics randomization."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_cube_a_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.15, -0.05), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cube_a"),
        },
    )

    reset_cube_b_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (0.05, 0.15), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cube_b"),
        },
    )

    # Physics randomization: mass for cube_a
    randomize_cube_a_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cube_a"),
            "mass_distribution_params": (0.05, 2.0),
            "operation": "abs",
        },
    )

    # Physics randomization: mass for cube_b
    randomize_cube_b_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cube_b"),
            "mass_distribution_params": (0.05, 2.0),
            "operation": "abs",
        },
    )

    # Physics randomization: friction for cube_a
    randomize_cube_a_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cube_a"),
            "static_friction_range": (0.2, 0.8),
            "dynamic_friction_range": (0.2, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # Physics randomization: friction for cube_b
    randomize_cube_b_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cube_b"),
            "static_friction_range": (0.2, 0.8),
            "dynamic_friction_range": (0.2, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for stacking — v2: added grasp reward to bridge reach→lift gap."""

    # 1. Reaching cube_a
    reaching_cube_a = RewTerm(func=cube_a_ee_distance, params={"std": 0.1}, weight=5.0)

    # 2. Grasping cube_a (gripper closed + close to cube) — KEY ADDITION
    grasping_cube_a = RewTerm(func=cube_a_grasp_reward, params={"std": 0.04}, weight=15.0)

    # 3. Lifting cube_a
    lifting_cube_a = RewTerm(func=cube_a_is_lifted, params={"scale": 0.08, "table_height": 0.03}, weight=30.0)

    # 4. Cube_a above cube_b (XY alignment, gated on lift)
    cube_a_above_b = RewTerm(func=cube_a_above_cube_b, params={"std": 0.03}, weight=20.0)

    # 5. Stacking success (full 3D alignment)
    stacking = RewTerm(
        func=stacking_success,
        params={"std": 0.02, "stack_height_offset": 0.06},
        weight=25.0,
    )

    # Action penalty (very small)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-6)

    # Joint velocity penalty
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-6,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for stacking."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    cube_a_dropping = DoneTerm(
        func=cube_a_dropping,
        params={"minimum_height": -0.05},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for stacking."""

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
class StackRLEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the stacking RL environment."""

    # Scene settings
    scene: StackSceneCfg = StackSceneCfg(num_envs=4096, env_spacing=2.5)
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

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

        # Set Franka as robot
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for Franka (joint position control)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )

        # Cube A: small red cube (0.05m) to be stacked on top
        self.scene.cube_a = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/CubeA",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, -0.1, 0.055], rot=[1, 0, 0, 0]),
            spawn=CuboidCfg(
                size=(0.05, 0.05, 0.05),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
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

        # Cube B: large blue cube (0.07m) — the base
        self.scene.cube_b = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/CubeB",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0.1, 0.055], rot=[1, 0, 0, 0]),
            spawn=CuboidCfg(
                size=(0.07, 0.07, 0.07),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
                mass_props=MassPropertiesCfg(density=600.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.8)),  # Blue
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
class StackRLEnvCfg_PLAY(StackRLEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
