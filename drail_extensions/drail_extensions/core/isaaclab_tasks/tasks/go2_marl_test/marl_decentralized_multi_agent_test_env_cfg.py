from typing import Type

import isaaclab.sim as sim_utils
import isaaclab_tasks.manager_based.locomotion.velocity.config.spot.mdp as spot_mdp
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    EventTermCfg as EventTerm,
    ManagerTermBaseCfg,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import drail_extensions.core.isaaclab_tasks.mdp as core_mdp
import drail_extensions.core.utils.math_utils as core_math_utils
from drail_extensions.core.isaaclab_robots.go2_robot import go2_robot
from drail_extensions.core.isaaclab_robots.go2_robot.go2_robot import GO2_JOINT_ORDER
from drail_extensions.core.isaaclab_rsl_rl.utils.vecenv_demarl_wrapper import (
    VecEnvDemarlWrapper,
)
from drail_extensions.core.managers import (
    RewardGroupCfg as RewGroup,
    TerminationGroupCfg as DoneGroup,
)

# Set agents far for apart for independent velocity tracking policy as they may collide with each other


def _update_term_params(term: ManagerTermBaseCfg, agent_id: int):
    if "asset_cfg" in term.params:
        term.params["asset_cfg"].name = f"robot_{agent_id}"
    if "command_name" in term.params:
        term.params["command_name"] = f"base_velocity_{agent_id}"
    if "sensor_cfg" in term.params:
        term.params["sensor_cfg"].name = f"contact_forces_{agent_id}"
    if "action_name" in term.params:
        term.params["action_name"] = f"joint_pos_{agent_id}"


##
# Scene definition
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=(
                f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/"
                "TilesMarbleSpiderWhiteBrickBondHoned.mdl"
            ),
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = go2_robot.GO2_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # # sensors
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    robot_1: ArticulationCfg = None
    contact_forces_1: ContactSensorCfg = None

    def __post_init__(self):
        self.robot_1 = self.robot.replace(prim_path="{ENV_REGEX_NS}/Robot_1")
        self.contact_forces_1 = self.contact_forces.replace(prim_path="{ENV_REGEX_NS}/Robot_1/.*")

        positions = core_math_utils.generate_grid_positions(n=2, spacing=20)

        self.robot.init_state.pos = (positions[0, 0].item(), positions[0, 1].item(), self.robot.init_state.pos[2])
        self.robot_1.init_state.pos = (positions[1, 0].item(), positions[1, 1].item(), self.robot_1.init_state.pos[2])


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = core_mdp.CategoricalVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 10.0),
        debug_vis=False,
        ranges={
            "stand": core_mdp.CategoricalVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.0, 0.0),
                lin_vel_y=(-0.0, 0.0),
                ang_vel_z=(-0.0, 0.0),
                probability=0.1,
            ),
            "in_place_turn": core_mdp.CategoricalVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.0, 0.0),
                lin_vel_y=(-0.0, 0.0),
                ang_vel_z=(-1.0, 1.0),
                probability=0.2,
            ),
            "walk_front_back": core_mdp.CategoricalVelocityCommandCfg.Ranges(
                lin_vel_x=(-1.0, 1.0),
                lin_vel_y=(-0.0, 0.0),
                ang_vel_z=(-0.0, 0.0),
                probability=0.35,
            ),
            "walk_sideways": core_mdp.CategoricalVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.0, 0.0),
                lin_vel_y=(-1.0, 1.0),
                ang_vel_z=(-0.0, 0.0),
                probability=0.1,
            ),
            "walk_front_back_turn": core_mdp.CategoricalVelocityCommandCfg.Ranges(
                lin_vel_x=(-1.0, 1.0),
                lin_vel_y=(-0.0, 0.0),
                ang_vel_z=(-1.0, 1.0),
                probability=0.1,
            ),
            "walk_sideways_turn": core_mdp.CategoricalVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.0, 0.0),
                lin_vel_y=(-1.0, 1.0),
                ang_vel_z=(-1.0, 1.0),
                probability=0.05,
            ),
            "all": core_mdp.CategoricalVelocityCommandCfg.Ranges(
                lin_vel_x=(-1.0, 1.0),
                lin_vel_y=(-1.0, 1.0),
                ang_vel_z=(-1.0, 1.0),
                probability=0.1,
            ),
        },
    )

    base_velocity_1 = None

    def __post_init__(self):
        self.base_velocity_1 = self.base_velocity.replace(asset_name="robot_1")


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = core_mdp.JointPositionAction_v2_Cfg(
        asset_name="robot",
        joint_names=go2_robot.GO2_JOINT_ORDER,
        scale=0.25,
        use_default_offset=True,
        preserve_order=True,
    )

    joint_pos_1 = None

    def __post_init__(self):
        self.joint_pos_1 = self.joint_pos.replace(asset_name="robot_1")


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(
            func=core_mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        projected_gravity = ObsTerm(
            func=core_mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        velocity_commands = ObsTerm(func=core_mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(
            func=core_mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=GO2_JOINT_ORDER, preserve_order=True)},
        )
        joint_vel = ObsTerm(
            func=core_mdp.joint_vel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=GO2_JOINT_ORDER, preserve_order=True)},
        )
        actions = ObsTerm(func=core_mdp.last_action, params={"action_name": "joint_pos"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        base_lin_vel = ObsTerm(
            func=core_mdp.base_lin_vel,
            noise=Unoise(n_min=-0.1, n_max=0.1),
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        base_ang_vel = ObsTerm(
            func=core_mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        projected_gravity = ObsTerm(
            func=core_mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        velocity_commands = ObsTerm(func=core_mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(
            func=core_mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=GO2_JOINT_ORDER, preserve_order=True)},
        )
        joint_vel = ObsTerm(
            func=core_mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=GO2_JOINT_ORDER, preserve_order=True)},
        )
        actions = ObsTerm(func=core_mdp.last_action, params={"action_name": "joint_pos"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()

    policy_1: PolicyCfg = None
    critic_1: CriticCfg = None

    def __post_init__(self):
        self.policy_1 = self.policy.replace()
        self.critic_1 = self.critic.replace()

        for term in vars(self.policy_1).values():
            if isinstance(term, ObsTerm):
                _update_term_params(term, 1)

        for term in vars(self.critic_1).values():
            if isinstance(term, ObsTerm):
                _update_term_params(term, 1)


@configclass
class EventCfg:
    """Configuration for events."""

    physics_material = EventTerm(
        func=core_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    base_external_force_torque = EventTerm(
        func=core_mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=(5.0, 10.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (-10.0, 10.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    random_joint_gains = EventTerm(
        func=core_mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=go2_robot.GO2_JOINT_ORDER, preserve_order=True),
            "stiffness_distribution_params": (0.75, 1.25),
            "damping_distribution_params": (0.75, 1.25),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    randomize_body_mass = EventTerm(
        func=core_mdp.randomize_rigid_body_mass,
        mode="reset",
        min_step_count_between_reset=0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.75, 1.25),
            "operation": "scale",
            "recompute_inertia": True,
        },
    )

    reset_base = EventTerm(
        func=core_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=core_mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=go2_robot.GO2_JOINT_ORDER, preserve_order=True),
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    push_robot = EventTerm(
        func=core_mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(3.0, 8.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    physics_material_1 = None
    base_external_force_torque_1 = None
    random_joint_gains_1 = None
    randomize_body_mass_1 = None
    reset_base_1 = None
    reset_robot_joints_1 = None
    push_robot_1 = None

    def __post_init__(self):
        self.physics_material_1 = self.physics_material.replace()
        _update_term_params(self.physics_material_1, 1)

        self.base_external_force_torque_1 = self.base_external_force_torque.replace()
        _update_term_params(self.base_external_force_torque_1, 1)

        self.random_joint_gains_1 = self.random_joint_gains.replace()
        _update_term_params(self.random_joint_gains_1, 1)

        self.randomize_body_mass_1 = self.randomize_body_mass.replace()
        _update_term_params(self.randomize_body_mass_1, 1)

        self.reset_base_1 = self.reset_base.replace()
        _update_term_params(self.reset_base_1, 1)

        self.reset_robot_joints_1 = self.reset_robot_joints.replace()
        _update_term_params(self.reset_robot_joints_1, 1)

        self.push_robot_1 = self.push_robot.replace()
        _update_term_params(self.push_robot_1, 1)


@configclass
class RewardsCfg:
    """Reward groups for the MDP."""

    @configclass
    class RewardsGroup(RewGroup):
        # Velocity tracking rewards
        base_linear_velocity = RewTerm(
            func=core_mdp.base_linear_velocity_reward_v2,
            weight=5.0,
            params={
                "std": 1.0,
                "ramp_rate": 0.5,
                "ramp_at_vel": 1.0,
                "asset_cfg": SceneEntityCfg("robot"),
                "command_name": "base_velocity",
            },
        )
        base_angular_velocity = RewTerm(
            func=core_mdp.base_angular_velocity_reward_v2,
            weight=5.0,
            params={"std": 2.0, "asset_cfg": SceneEntityCfg("robot"), "command_name": "base_velocity"},
        )
        # Gait rewards
        feet_air_time = RewTerm(
            func=core_mdp.feet_air_time_v2,
            weight=5.0,
            params={
                "command_name": "base_velocity",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot"),
                "threshold": 1.0,
            },
        )
        gait = RewTerm(
            func=core_mdp.GaitReward_v2,
            weight=5.0,
            params={
                "std": 0.1,
                "max_err": 0.2,
                "synced_feet_pair_names": (("FL_foot", "RR_foot"), ("FR_foot", "RL_foot")),
                "asset_cfg": SceneEntityCfg("robot"),
                "sensor_cfg": SceneEntityCfg("contact_forces"),
                "command_name": "base_velocity",
            },
        )
        foot_clearance = RewTerm(
            func=core_mdp.foot_clearance_reward_v2,
            weight=0.5,
            params={
                "std": 0.05,
                "tanh_mult": 2.0,
                "target_height": 0.06,
                "command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            },
        )
        foot_slip = RewTerm(
            func=spot_mdp.foot_slip_penalty,
            weight=-0.5,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
                "threshold": 1.0,
            },
        )
        feet_contact = RewTerm(
            func=core_mdp.feet_contact_quadruped,
            weight=5.0,
            params={
                "command_name": "base_velocity",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot"),
            },
        )
        # Penalties
        action_smoothness = RewTerm(
            func=core_mdp.action_smoothness_penalty_v2, weight=-1.0, params={"action_name": "joint_pos"}
        )
        air_time_variance = RewTerm(
            func=core_mdp.air_time_variance_penalty_v2,
            weight=-1.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
                "command_name": "base_velocity",
            },
        )
        base_motion = RewTerm(
            func=spot_mdp.base_motion_penalty, weight=-2.0, params={"asset_cfg": SceneEntityCfg("robot")}
        )
        base_orientation = RewTerm(
            func=spot_mdp.base_orientation_penalty, weight=-3.0, params={"asset_cfg": SceneEntityCfg("robot")}
        )
        joint_acc = RewTerm(
            func=spot_mdp.joint_acceleration_penalty,
            weight=-1.0e-4,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
        )
        joint_pos = RewTerm(
            func=core_mdp.joint_position_penalty_v2,
            weight=-0.7,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "stand_still_scale": 5.0,
                "velocity_threshold": 0.5,
                "command_name": "base_velocity",
            },
        )
        joint_torques = RewTerm(
            func=spot_mdp.joint_torques_penalty,
            weight=-5.0e-4,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
        )
        joint_vel = RewTerm(
            func=spot_mdp.joint_velocity_penalty,
            weight=-1.0e-2,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
        )

    reward: RewardsGroup = RewardsGroup()
    reward_1: RewardsGroup = None

    def __post_init__(self):
        self.reward_1 = self.reward.replace()
        for term in vars(self.reward_1).values():
            if isinstance(term, RewTerm):
                _update_term_params(term, 1)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    @configclass
    class TerminationGroup(DoneGroup):
        time_out = DoneTerm(func=core_mdp.time_out, time_out=True)
        base_contact = DoneTerm(
            func=core_mdp.illegal_contact,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
        )
        base_height = DoneTerm(
            func=core_mdp.base_height_out_of_range, params={"asset_cfg": SceneEntityCfg("robot"), "min_height": 0.3}
        )

    done: TerminationGroup = TerminationGroup()

    done_1: TerminationGroup = None

    def __post_init__(self):
        self.done_1 = self.done.replace()
        for term in vars(self.done_1).values():
            if isinstance(term, DoneTerm):
                _update_term_params(term, 1)


# Environment configuration
##


@configclass
class Go2_MARL_Decentralized_Multi_Agent_Test_EnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    wrapper_class: Type[VecEnvDemarlWrapper] = VecEnvDemarlWrapper
    num_agents: int = 2

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        if self.scene.contact_forces_1 is not None:
            self.scene.contact_forces_1.update_period = self.sim.dt


@configclass
class Go2_MARL_Decentralized_Multi_Agent_Test_EnvCfg_PLAY(Go2_MARL_Decentralized_Multi_Agent_Test_EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 5000.0

        self.scene.contact_forces.debug_vis = True
        self.scene.contact_forces_1.debug_vis = True

        # Bring agents closer for visualization
        positions = core_math_utils.generate_grid_positions(n=2, spacing=2.5)
        self.scene.robot.init_state.pos = (
            positions[0, 0].item(),
            positions[0, 1].item(),
            self.scene.robot.init_state.pos[2],
        )
        self.scene.robot_1.init_state.pos = (
            positions[1, 0].item(),
            positions[1, 1].item(),
            self.scene.robot_1.init_state.pos[2],
        )

        self.commands.base_velocity = core_mdp.GUIKeyboardVelocityCommandCfg(
            asset_name="robot",
            debug_vis=True,
        )
        self.commands.base_velocity_1 = core_mdp.GUIKeyboardVelocityCommandCfg(
            asset_name="robot_1",
            debug_vis=True,
        )

        self.events.push_robot = None
        self.events.push_robot_1 = None
        self.events.base_external_force_torque = None
        self.events.base_external_force_torque_1 = None

        self.observations.policy.enable_corruption = False
        self.observations.policy_1.enable_corruption = False
