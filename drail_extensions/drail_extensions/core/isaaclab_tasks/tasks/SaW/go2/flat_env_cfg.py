# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    EventTermCfg as EventTerm,
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

import drail_extensions.core.isaaclab_tasks.mdp as mdp

##
# Pre-defined configs
##
from drail_extensions.core.isaaclab_robots.go2_robot import GO2_ROBOT_CFG

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
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",  # noqa: E501
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = GO2_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

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
    # marker_light = AssetBaseCfg(
    #     prim_path="/World/envs/env_0/Robot/base/marker_sphere",
    #     spawn=sim_utils.SphereCfg(
    #         radius=0.1,
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
    #     ),
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.2), rot=(1.0, 0.0, 0.0, 0.0)),
    # )


##
# MDP settings
##


#########################################################
#### Interactive Commands #### # noqa: E266
#########################################################

import isaaclab.utils.math as math_utils  # noqa: E402
import omni  # noqa: E402
import torch  # noqa: E402

# if TYPE_CHECKING:
from isaaclab.envs import ManagerBasedEnv  # noqa: E402


class UniformVelocityCommand_Interactive(mdp.UniformVelocityCommand):
    def __init__(self, cfg: mdp.UniformVelocityCommandCfg, env: ManagerBasedEnv):

        #########################################################
        # Keyboard Configuration
        #########################################################
        self.keyboard_cfg = {
            "speeds": {
                "forward": 3 * cfg.ranges.lin_vel_x[1] / 4,
                "backward": 3 * cfg.ranges.lin_vel_x[0] / 4,
                "side_left": 3 * cfg.ranges.lin_vel_y[0] / 4,
                "side_right": 3 * cfg.ranges.lin_vel_y[1] / 4,
            },
            "heading_angles": {
                "forward": 0.0,  # 0 radians
                "backward": 3.14159,  # π radians
                "left": 1.57,  # π/2 radians
                "right": -1.57,  # -π/2 radians
            },
        }

        self._activate_keyboard = False  # Whether to activate keyboard input
        self._keyboard_initialized = False  # Whether keyboard input is initialized
        self._initialize_keyboard()
        #########################################################
        # End of Keyboard Configuration
        #########################################################

        super().__init__(cfg, env)

    def _initialize_keyboard(self):
        from drail_extensions.core.utils.keyboard import RawKeyboard

        try:
            self.keyboard = RawKeyboard()
            self._keyboard_initialized = True
            omni.log.info("Successfully initialized keyboard controls")
        except ImportError:
            omni.log.warn("Could not initialize keyboard controls.")
            self._keyboard_initialized = False

    def _activate_keyboard(self):
        if not self._keyboard_initialized:
            omni.log.warn("Keyboard controls not initialized. Try: pip install pynput")
            return
        self._activate_keyboard = True

    def _deactivate_keyboard(self):
        if not self._keyboard_initialized:
            omni.log.warn("Keyboard controls not initialized. Try: pip install pynput")
            return
        self._activate_keyboard = False

    def _update_command(self):
        super()._update_command()

        if self._activate_keyboard:
            if self._keyboard_initialized:
                # Reset commands
                self.vel_command_b[:] = 0.0
                self.heading_target[:] = 0.0

                # Update velocity commands
                if self.keyboard.get_input() == "up":
                    self.vel_command_b[:, 0] = self.keyboard_cfg["speeds"]["forward"]
                elif self.keyboard.get_input() == "down":
                    self.vel_command_b[:, 0] = self.keyboard_cfg["speeds"]["backward"]
                elif self.keyboard.get_input() == "left":
                    self.vel_command_b[:, 1] = self.keyboard_cfg["speeds"]["side_left"]
                elif self.keyboard.get_input() == "right":
                    self.vel_command_b[:, 1] = self.keyboard_cfg["speeds"]["side_right"]

                # Update heading targets
                if self.keyboard.get_input() == "w":
                    self.heading_target[:] = self.keyboard_cfg["heading_angles"]["forward"]
                elif self.keyboard.get_input() == "s":
                    self.heading_target[:] = self.keyboard_cfg["heading_angles"]["backward"]
                elif self.keyboard.get_input() == "a":
                    self.heading_target[:] = self.keyboard_cfg["heading_angles"]["left"]
                elif self.keyboard.get_input() == "d":
                    self.heading_target[:] = self.keyboard_cfg["heading_angles"]["right"]

                #########################################################
                # Compute angular velocity from heading direction
                # Copied from the origintal update command function
                #########################################################
                if self.cfg.heading_command:
                    # resolve indices of heading envs
                    env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
                    # compute angular velocity
                    heading_error = math_utils.wrap_to_pi(
                        self.heading_target[env_ids] - self.robot.data.heading_w[env_ids]
                    )
                    self.vel_command_b[env_ids, 2] = torch.clip(
                        self.cfg.heading_control_stiffness * heading_error,
                        min=self.cfg.ranges.ang_vel_z[0],
                        max=self.cfg.ranges.ang_vel_z[1],
                    )
                #########################################################
                # End of Copied from the origintal update command function
                #########################################################
            else:
                omni.log.warn("Keyboard controls not initialized. Try: pip install pynput")


@configclass
class UniformVelocityInteractiveCommandCfg(mdp.UniformVelocityCommandCfg):
    """Configuration for the uniform velocity command generator."""

    class_type: type = UniformVelocityCommand_Interactive


#########################################################
#########################################################


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = UniformVelocityInteractiveCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=UniformVelocityInteractiveCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
    )


JOINT_ORDER = [
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
]


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=JOINT_ORDER, scale=0.25, use_default_offset=True, preserve_order=True
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_ORDER, preserve_order=True)},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_ORDER, preserve_order=True)},
        )
        actions = ObsTerm(func=mdp.last_action, params={"action_name": "joint_pos"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_ORDER, preserve_order=True)},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_ORDER, preserve_order=True)},
        )
        actions = ObsTerm(func=mdp.last_action, params={"action_name": "joint_pos"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 1.9),
            "dynamic_friction_range": (0.6, 1.5),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=(5.0, 10.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (-10.0, 10.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    random_joint_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_ORDER, preserve_order=True),
            "stiffness_distribution_params": (0.75, 1.25),
            "damping_distribution_params": (0.75, 1.25),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    randomize_body_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        min_step_count_between_reset=0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.75, 1.25),
            "operation": "scale",
            "recompute_inertia": True,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
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
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_ORDER, preserve_order=True),
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
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


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25), "asset_cfg": SceneEntityCfg("robot")},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.75,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25), "asset_cfg": SceneEntityCfg("robot")},
    )
    # -- penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0, params={"asset_cfg": SceneEntityCfg("robot")})
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05, params={"asset_cfg": SceneEntityCfg("robot")})
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-0.0002, params={"asset_cfg": SceneEntityCfg("robot")})
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7, params={"asset_cfg": SceneEntityCfg("robot")})
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot"),
            "threshold": 1.0,
        },
    )
    # undesired_contacts = RewTerm(
    #     func=mdp.undesired_contacts,
    #     weight=0.0,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*thigh"), "threshold": 1.0},
    # )
    # -- optional penalties
    flat_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2, weight=-2.5, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_ORDER, preserve_order=True)},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
        # params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base", ".*thigh", ".*calf"]), "threshold": 1.0}, # noqa: E501
    )
    base_height = DoneTerm(
        func=mdp.base_height_out_of_range, params={"asset_cfg": SceneEntityCfg("robot"), "min_height": 0.28}
    )


# Environment configuration
##


@configclass
class Go2FlatEnvCfg(ManagerBasedRLEnvCfg):
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


@configclass
class Go2FlatEnvCfg_PLAY(Go2FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 5.0

        self.scene.contact_forces.debug_vis = True

        # Enable keyboard for interactive control
        self.commands.base_velocity = mdp.GUIKeyboardVelocityCommandCfg(
            asset_name="robot",
            debug_vis=True,
        )
