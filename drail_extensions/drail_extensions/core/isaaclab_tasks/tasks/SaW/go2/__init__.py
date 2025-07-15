# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import flat_env_cfg, rsl_rl_ppo_cfg, spot_reward_env_cfg

##
# Register Gym environments.
##

# AMP-based velocity control
gym.register(
    id="SaW-Go2-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.Go2FlatEnvCfg,
        "rsl_rl_cfg_entry_point": rsl_rl_ppo_cfg.Go2FlatPPORunnerCfg,
    },
)


gym.register(
    id="SaW-Go2-Flat-v0-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.Go2FlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": rsl_rl_ppo_cfg.Go2FlatPPORunnerCfg,
    },
)


# Spot reward on Go2 for velocity control
gym.register(
    id="SaW-Go2-Velocity-Spot-Reward-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": spot_reward_env_cfg.Go2_Velocity_SpotReward_EnvCfg,
        "rsl_rl_cfg_entry_point": rsl_rl_ppo_cfg.Go2_Velocity_SpotReward_RunnerCfg,
    },
)


gym.register(
    id="SaW-Go2-Velocity-Spot-Reward-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": spot_reward_env_cfg.Go2_Velocity_SpotReward_EnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": rsl_rl_ppo_cfg.Go2_Velocity_SpotReward_RunnerCfg,
    },
)
