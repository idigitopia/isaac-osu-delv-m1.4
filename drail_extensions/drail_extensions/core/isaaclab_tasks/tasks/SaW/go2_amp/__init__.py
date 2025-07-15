# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import flat_amp_env_cfg, rsl_rl_amp_ppo_cfg

##
# Register Gym environments.
##

# AMP-based velocity control
gym.register(
    id="SaW-Go2-Flat-AMP-Penalty-FeetAirTime-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_amp_env_cfg.Go2_Task_AMP_FeetAirTime_Penalty_FlatEnvCfg,
        "rsl_rl_cfg_entry_point": rsl_rl_amp_ppo_cfg.Go2AMPFlatPPORunnerCfg,
    },
)


gym.register(
    id="SaW-Go2-Flat-AMP-Penalty-FeetAirTime-v0-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_amp_env_cfg.Go2_AMP_FlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": rsl_rl_amp_ppo_cfg.Go2AMPFlatPPORunnerCfg,
    },
)
