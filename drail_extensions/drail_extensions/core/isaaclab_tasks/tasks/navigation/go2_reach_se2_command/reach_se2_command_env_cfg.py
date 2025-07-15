import math

from isaaclab.managers import (
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
)
from isaaclab.markers.config import GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import drail_extensions.core.isaaclab_tasks.mdp as mdp
import drail_extensions.core.isaaclab_tasks.tasks.SaW.go2.spot_reward_env_cfg as spot_reward_env_cfg

LOW_LEVEL_ENV_CFG = spot_reward_env_cfg.Go2_Velocity_SpotReward_EnvCfg()


@configclass
class MySceneCfg(spot_reward_env_cfg.MySceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    def __post_init__(self):
        super().__post_init__()
        self.contact_forces.track_air_time = False


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    pose_command = mdp.UniformPose2dCommandCfg(
        asset_name="robot",
        simple_heading=False,
        resampling_time_range=(8.0, 8.0),
        debug_vis=True,
        ranges=mdp.UniformPose2dCommandCfg.Ranges(pos_x=(-3.0, 3.0), pos_y=(-3.0, 3.0), heading=(-math.pi, math.pi)),
        goal_pose_visualizer_cfg=GREEN_ARROW_X_MARKER_CFG.replace(
            prim_path="/Visuals/Command/pose_goal",
        ),
    )

    def __post_init__(self):
        self.pose_command.goal_pose_visualizer_cfg.markers["arrow"].visual_material.diffuse_color = (1.0, 1.0, 0.0)
        self.pose_command.goal_pose_visualizer_cfg.markers["arrow"].scale = (0.2, 0.2, 0.8)


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    pre_trained_policy_action: mdp.PreTrainedPolicyActionCfg = mdp.PreTrainedPolicyActionCfg(
        asset_name="robot",
        policy_path=(
            "drail_extensions/drail_extensions/core/data/pretrained_policy/"
            "SaW-Go2-Velocity-Spot-Reward-Play-v0/team-osu/isaaclab/gkmo1h09/exported/policy.pt"
        ),
        low_level_decimation=4,
        low_level_actions=LOW_LEVEL_ENV_CFG.actions.joint_pos,
        low_level_observations=LOW_LEVEL_ENV_CFG.observations.policy,
        high_level_actions_clipping=[(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)],
        low_level_observations_command_name="velocity_commands",
        action_threshold=0.1,
        action_dim=3,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2), params={"asset_cfg": SceneEntityCfg("robot")}
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "pose_command"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1), params={"asset_cfg": SceneEntityCfg("robot")}
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2), params={"asset_cfg": SceneEntityCfg("robot")}
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "pose_command"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    # termination_penalty = RewTerm(func=mdp.is_terminated, weight=-400.0)
    position_tracking = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.5,
        params={"std": 2.0, "command_name": "pose_command"},
    )
    position_tracking_fine_grained = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.5,
        params={"std": 0.2, "command_name": "pose_command"},
    )
    heading_tracking = RewTerm(
        func=mdp.heading_command_error_abs,
        weight=-0.2,
        params={"command_name": "pose_command"},
    )


# Environment configuration
##


@configclass
class Go2_Reach_SE2_Command_EnvCfg(spot_reward_env_cfg.Go2_Velocity_SpotReward_EnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        """Post initialization."""
        # general settings
        self.decimation *= 10  # Navigation control runs 10x slower than low-level velocity control (This is 5hz)
        self.sim.render_interval = self.decimation
        self.episode_length_s = self.commands.pose_command.resampling_time_range[1]


@configclass
class Go2_Reach_SE2_Command_EnvCfg_PLAY(Go2_Reach_SE2_Command_EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        # Render at same rate as low-level velocity control for smoother visualizations
        self.sim.render_interval = LOW_LEVEL_ENV_CFG.sim.render_interval

        self.scene.contact_forces.debug_vis = True
