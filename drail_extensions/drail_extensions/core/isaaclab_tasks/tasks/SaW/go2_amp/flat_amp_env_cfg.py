# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.managers import (
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
)
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import drail_extensions.core.isaaclab_tasks.mdp as mdp
from drail_extensions.core.isaaclab_tasks.tasks.SaW.go2.flat_env_cfg import (
    JOINT_ORDER,
    Go2FlatEnvCfg,
    ObservationsCfg,
)

##
# Pre-defined configs

##
# Scene definition
##


@configclass
class AMPObservationsCfg(ObservationsCfg):
    @configclass
    class AMPCfg(ObsGroup):
        """Observations for critic group."""

        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_ORDER, preserve_order=True)},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_ORDER, preserve_order=True)},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 10

    amp = AMPCfg()


#########
@configclass
class Rewards_SaW_Task_Cfg:
    """Reward terms for the MDP."""

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


@configclass
class Rewards_SaW_Task_AMP_Cfg(Rewards_SaW_Task_Cfg):
    style_reward = RewTerm(func=mdp.style_reward, weight=0.4, params={"amp_observation_key": "amp"})


@configclass
class Rewards_SaW_Task_AMP_FeetAirTime_Cfg(Rewards_SaW_Task_AMP_Cfg):
    """Reward terms for the MDP."""

    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot"),
            "threshold": 1.0,
        },
    )


@configclass
class Rewards_SaW_Task_AMP_FeetAirTime_Penalty_Cfg(Rewards_SaW_Task_AMP_FeetAirTime_Cfg):
    """Reward terms for the MDP."""

    # -- penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0, params={"asset_cfg": SceneEntityCfg("robot")})
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05, params={"asset_cfg": SceneEntityCfg("robot")})
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-0.0002, params={"asset_cfg": SceneEntityCfg("robot")})
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7, params={"asset_cfg": SceneEntityCfg("robot")})
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    # -- optional penalties
    flat_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2, weight=-2.5, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_ORDER, preserve_order=True)},
    )

    def __post_init__(self):
        pass
        # self.style_reward.weight = 1.0


from drail_extensions.core.isaaclab_tasks.tasks.SaW.go2.flat_env_cfg import (  # noqa: E402
    TerminationsCfg,
)


@configclass
class AMPTerminationsCfg(TerminationsCfg):
    base_height = None


######
# Environment configuration
######


@configclass
class Go2_AMP_FlatEnvCfg(Go2FlatEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    observations: AMPObservationsCfg = AMPObservationsCfg()
    # MDP settings
    rewards: Rewards_SaW_Task_AMP_Cfg = Rewards_SaW_Task_AMP_Cfg()
    terminations: AMPTerminationsCfg = AMPTerminationsCfg()


@configclass
class Go2_AMP_FeetAirTime_FlatEnvCfg(Go2_AMP_FlatEnvCfg):
    rewards: Rewards_SaW_Task_AMP_FeetAirTime_Cfg = Rewards_SaW_Task_AMP_FeetAirTime_Cfg()


@configclass
class Go2_Task_AMP_FeetAirTime_Penalty_FlatEnvCfg(Go2_AMP_FeetAirTime_FlatEnvCfg):
    rewards: Rewards_SaW_Task_AMP_FeetAirTime_Penalty_Cfg = Rewards_SaW_Task_AMP_FeetAirTime_Penalty_Cfg()


@configclass
class Go2_AMP_FlatEnvCfg_PLAY(Go2_AMP_FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # Prevent from needing discriminator model to compute style reward in play mode
        self.rewards.style_reward.weight = 0.0

        self.scene.contact_forces.debug_vis = True

        # Enable keyboard for interactive control
        self.commands.base_velocity = mdp.GUIKeyboardVelocityCommandCfg(
            asset_name="robot",
            debug_vis=True,
        )
