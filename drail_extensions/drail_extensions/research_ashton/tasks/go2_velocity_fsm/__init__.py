# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import (
    rsl_rl_velocity_ppo_cfg,
    pose_wrapper_env_cfg,
)

##
# Register Gym environments.
##


# Pose wrapper for spot reward velocity env
gym.register(
    id="ashton-Go2-Velocity-FSM-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pose_wrapper_env_cfg.Go2PoseWrapperEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": rsl_rl_velocity_ppo_cfg.Go2PoseWrapperPPORunnerCfg,
    },
)
