"""PhysREPA Pick-and-Place environment: Franka picks cube and places at target position."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from physrepa_tasks import mdp
from physrepa_tasks.envs.lift_env_cfg import ActionsCfg, PhysREPALiftSceneCfg  # ActionsCfg now uses IK relative


##
# Scene definition
##


@configclass
class PhysREPAPickPlaceSceneCfg(PhysREPALiftSceneCfg):
    """Scene with Franka robot, object, table, cameras, contact sensor, and target marker."""

    # Visual target marker — bright green disc on table
    target_marker = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TargetMarker",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0.0, 0.005]),
        spawn=sim_utils.CylinderCfg(
            radius=0.04,
            height=0.005,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        ),
    )


##
# MDP settings
##


@configclass
class ObservationsCfg:
    """Observation specifications for Pick-and-Place."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observations (state + image)."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        target_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
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
        object_position = ObsTerm(func=mdp.object_position_w)
        object_orientation = ObsTerm(func=mdp.object_orientation_w)
        object_velocity = ObsTerm(func=mdp.object_velocity_w)
        object_angular_velocity = ObsTerm(func=mdp.object_angular_velocity_w)
        ee_to_object_distance = ObsTerm(func=mdp.ee_to_object_distance)
        contact_flag = ObsTerm(func=mdp.contact_flag)
        contact_force = ObsTerm(func=mdp.contact_force)
        target_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    physics_gt: PhysicsGTCfg = PhysicsGTCfg()


@configclass
class EventCfg:
    """Event configuration: resets + physics randomization."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # Sync marker to command target position
#    sync_marker = EventTerm(
#        func=mdp.sync_target_marker,
#        mode="reset",
#        params={"command_name": "object_pose", "marker_name": "target_marker"},
#    )

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.20, 0.20), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )

    # Physics randomization: object mass
    randomize_object_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (0.05, 2.0),
            "operation": "abs",
        },
    )

    # Physics randomization: object friction
    randomize_object_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": (0.2, 0.8),
            "dynamic_friction_range": (0.2, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
            "make_consistent": True,
        },
    )



@configclass
class RewardsCfg:
    """Reward terms for Pick-and-Place."""

    # Phase 1: reach the object
    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.1}, weight=1.0)

    # Phase 2: lift the object
    lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.04}, weight=10.0)

    # Phase 3: move object to target place position (using command-based goal tracking)
    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.3, "minimal_height": 0.04, "command_name": "object_pose"},
        weight=16.0,
    )

    # Phase 3 fine: precise placement
    place_success = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.05, "minimal_height": 0.02, "command_name": "object_pose"},
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
    """Termination terms for Pick-and-Place."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")}
    )


@configclass
class CommandsCfg:
    """Command terms: target place position sampled on table surface."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="panda_hand",
        resampling_time_range=(22.0, 22.0),  # longer than episode to prevent resample
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.35, 0.65), pos_y=(-0.20, 0.20), pos_z=(0.02, 0.02),
            roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0),
        ),
    )


##
# Environment configuration
##


@configclass
class PhysREPAPickPlaceEnvCfg(ManagerBasedRLEnvCfg):
    """PhysREPA Pick-and-Place environment with cameras, contact sensor, and physics randomization."""

    # Scene
    scene: PhysREPAPickPlaceSceneCfg = PhysREPAPickPlaceSceneCfg(num_envs=16, env_spacing=5.0)

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
