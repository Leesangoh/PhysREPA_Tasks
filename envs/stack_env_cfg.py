"""PhysREPA Stack environment: Franka picks cube_a and stacks it on top of cube_b."""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg, DifferentialInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

from physrepa_tasks import mdp
from physrepa_tasks.envs.lift_env_cfg import ActionsCfg


##
# Scene definition
##


@configclass
class PhysREPAStackSceneCfg(InteractiveSceneCfg):
    """Scene with Franka robot, two cubes, table, cameras, and contact sensor."""

    # Robot (with contact sensors enabled on gripper fingers)
    robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=FRANKA_PANDA_HIGH_PD_CFG.spawn.replace(activate_contact_sensors=True),
    )

    # Cube A: the small cube to be picked up and placed on top of cube B
    cube_a: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CubeA",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, -0.1, 0.035], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(
            size=(0.06, 0.06, 0.06),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.08),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.1, 0.1)),  # red
            activate_contact_sensors=True,
        ),
    )

    # Cube B: the large base cube that stays on the table
    cube_b: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CubeB",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0.1, 0.04], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(
            size=(0.07, 0.07, 0.07),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.15),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.2, 0.9)),  # blue
            activate_contact_sensors=True,
        ),
    )

    # End-effector frame sensor
    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
        debug_vis=False,
        visualizer_cfg=FRAME_MARKER_CFG.replace(prim_path="/Visuals/FrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                name="end_effector",
                offset=OffsetCfg(pos=[0.0, 0.0, 0.1034]),
            ),
        ],
    )

    # Contact sensor on gripper fingers — filter on both cubes
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
        update_period=0.0,
        history_length=1,
        track_air_time=True,
        track_contact_points=True,
        force_threshold=0.5,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/CubeA", "{ENV_REGEX_NS}/CubeB"],
    )

    # Contact sensor for cube_a <-> cube_b contact detection
    cube_contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/CubeA",
        update_period=0.0,
        history_length=1,
        track_air_time=True,
        track_contact_points=True,
        force_threshold=0.5,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/CubeB"],
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

    # Lighting
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # 3rd-person camera (front-side view) — 256x256 for V-JEPA 2
    table_cam: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/table_cam",
        update_period=0.0,
        height=384,
        width=384,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=19.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 4.0)
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(1.6, 0.0, 0.90), rot=(0.33900, -0.62054, -0.62054, 0.33900), convention="ros"
        ),
    )

    # Wrist camera — 256x256
    wrist_cam: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam",
        update_period=0.0,
        height=384,
        width=384,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 2.0)
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.13, 0.0, -0.15), rot=(-0.70614, 0.03701, 0.03701, -0.70614), convention="ros"
        ),
    )


##
# MDP settings
##


@configclass
class ObservationsCfg:
    """Observation specifications for Stack."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observations (state + image)."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        cube_a_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={"object_cfg": SceneEntityCfg("cube_a")},
        )
        cube_b_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={"object_cfg": SceneEntityCfg("cube_b")},
        )
        actions = ObsTerm(func=mdp.last_action)

        # Camera observations
        table_cam = ObsTerm(
            func=mdp.image, params={"sensor_cfg": SceneEntityCfg("table_cam"), "data_type": "rgb", "normalize": False}
        )
        wrist_cam = ObsTerm(
            func=mdp.image, params={"sensor_cfg": SceneEntityCfg("wrist_cam"), "data_type": "rgb", "normalize": False}
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class PhysicsGTCfg(ObsGroup):
        """Physics ground truth observations (for probing, not policy input)."""

        ee_position = ObsTerm(func=mdp.ee_position_w)
        ee_position_b = ObsTerm(func=mdp.ee_position_b)
        ee_velocity = ObsTerm(func=mdp.ee_velocity_w)
        cube_a_position = ObsTerm(
            func=mdp.object_position_w, params={"object_cfg": SceneEntityCfg("cube_a")}
        )
        cube_a_orientation = ObsTerm(
            func=mdp.object_orientation_w, params={"object_cfg": SceneEntityCfg("cube_a")}
        )
        cube_a_velocity = ObsTerm(
            func=mdp.object_velocity_w, params={"object_cfg": SceneEntityCfg("cube_a")}
        )
        cube_a_angular_velocity = ObsTerm(
            func=mdp.object_angular_velocity_w, params={"object_cfg": SceneEntityCfg("cube_a")}
        )
        cube_b_position = ObsTerm(
            func=mdp.object_position_w, params={"object_cfg": SceneEntityCfg("cube_b")}
        )
        cube_b_orientation = ObsTerm(
            func=mdp.object_orientation_w, params={"object_cfg": SceneEntityCfg("cube_b")}
        )
        cube_b_velocity = ObsTerm(
            func=mdp.object_velocity_w, params={"object_cfg": SceneEntityCfg("cube_b")}
        )
        cube_b_angular_velocity = ObsTerm(
            func=mdp.object_angular_velocity_w, params={"object_cfg": SceneEntityCfg("cube_b")}
        )
        ee_to_cube_a_distance = ObsTerm(
            func=mdp.ee_to_object_distance, params={"object_cfg": SceneEntityCfg("cube_a")}
        )
        ee_to_cube_b_distance = ObsTerm(
            func=mdp.ee_to_object_distance, params={"object_cfg": SceneEntityCfg("cube_b")}
        )
        cube_a_to_cube_b_distance = ObsTerm(
            func=mdp.object_object_distance,
            params={"object_a_cfg": SceneEntityCfg("cube_a"), "object_b_cfg": SceneEntityCfg("cube_b")},
        )
        contact_flag = ObsTerm(func=mdp.contact_flag)
        contact_force = ObsTerm(func=mdp.contact_force)
        object_object_contact_flag = ObsTerm(
            func=mdp.contact_flag, params={"sensor_cfg": SceneEntityCfg("cube_contact_sensor")}
        )
        object_object_contact_force = ObsTerm(
            func=mdp.contact_force, params={"sensor_cfg": SceneEntityCfg("cube_contact_sensor")}
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    physics_gt: PhysicsGTCfg = PhysicsGTCfg()


@configclass
class EventCfg:
    """Event configuration: resets + physics randomization for both cubes."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # Randomize cube_a position (the one to be picked)
    reset_cube_a_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.25, -0.05), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cube_a", body_names="CubeA"),
        },
    )

    # Randomize cube_b position (the base cube)
    reset_cube_b_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (0.05, 0.25), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cube_b", body_names="CubeB"),
        },
    )

    # Physics randomization: cube_a mass
    randomize_cube_a_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cube_a"),
            "mass_distribution_params": (0.05, 2.0),
            "operation": "abs",
        },
    )

    # Physics randomization: cube_b mass
    randomize_cube_b_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cube_b"),
            "mass_distribution_params": (0.05, 2.0),
            "operation": "abs",
        },
    )

    # Physics randomization: cube_a friction
    randomize_cube_a_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cube_a"),
            "static_friction_range": (0.2, 0.8),
            "dynamic_friction_range": (0.2, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
            "make_consistent": True,
        },
    )

    # Physics randomization: cube_b friction
    randomize_cube_b_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cube_b"),
            "static_friction_range": (0.2, 0.8),
            "dynamic_friction_range": (0.2, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
            "make_consistent": True,
        },
    )



@configclass
class RewardsCfg:
    """Reward terms for Stack.

    Rewards guide the robot through: reach cube_a -> lift cube_a -> position above cube_b -> stack.
    """

    # Phase 1: EE reaches cube_a
    reaching_cube_a = RewTerm(
        func=mdp.object_ee_distance,
        params={"std": 0.1, "object_cfg": SceneEntityCfg("cube_a")},
        weight=1.0,
    )

    # Phase 2: lift cube_a
    lifting_cube_a = RewTerm(
        func=mdp.object_is_lifted,
        params={"minimal_height": 0.04, "object_cfg": SceneEntityCfg("cube_a")},
        weight=15.0,
    )

    # Phase 3: cube_a above cube_b (use object_goal_distance with command pointing above cube_b)
    # We use the command manager to generate a target that is above cube_b's position.
    # The command is set to sample above the table at stacking height.
    cube_a_above_b = RewTerm(
        func=mdp.object_goal_distance,
        params={
            "std": 0.3,
            "minimal_height": 0.04,
            "command_name": "object_pose",
            "object_cfg": SceneEntityCfg("cube_a"),
        },
        weight=16.0,
    )

    # Phase 4: stacking success (fine tracking)
    stacking_success = RewTerm(
        func=mdp.object_goal_distance,
        params={
            "std": 0.05,
            "minimal_height": 0.04,
            "command_name": "object_pose",
            "object_cfg": SceneEntityCfg("cube_a"),
        },
        weight=10.0,
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for Stack."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    cube_a_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("cube_a")},
    )
    cube_b_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("cube_b")},
    )


@configclass
class CommandsCfg:
    """Command terms: target stacking position (above cube_b's initial position).

    The target z is set to the stacking height (~0.11m = cube_b table height 0.055 + cube size).
    Position x/y ranges overlap with cube_b's randomized position range.
    """

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="panda_hand",
        resampling_time_range=(22.0, 22.0),
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.4, 0.6), pos_y=(0.05, 0.25), pos_z=(0.11, 0.11),
            roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0),
        ),
    )


##
# Environment configuration
##


@configclass
class PhysREPAStackEnvCfg(ManagerBasedRLEnvCfg):
    """PhysREPA Stack environment with cameras, contact sensor, and physics randomization."""

    # Scene
    scene: PhysREPAStackSceneCfg = PhysREPAStackSceneCfg(num_envs=16, env_spacing=5.0)

    # MDP
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    # Camera rendering
    rerender_on_reset = True

    def __post_init__(self):
        """Post initialization."""
        self.decimation = 2
        self.episode_length_s = 20.0
        self.sim.dt = 0.01  # 100Hz physics
        self.sim.render_interval = self.decimation
        self.sim.render.antialiasing_mode = "OFF"

        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
