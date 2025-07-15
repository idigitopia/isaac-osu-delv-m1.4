import gymnasium as gym

from . import reach_se2_box_env_cfg, rsl_rl_reach_se2_box_ppo_cfg

##
# Register Gym environments.
##


gym.register(
    id="Go2-Reach-SE2-Box-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": reach_se2_box_env_cfg.Go2_Reach_SE2_Box_EnvCfg,
        "rsl_rl_cfg_entry_point": rsl_rl_reach_se2_box_ppo_cfg.Go2_Reach_SE2_Box_PPORunnerCfg,
    },
)

gym.register(
    id="Go2-Reach-SE2-Box-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": reach_se2_box_env_cfg.Go2_Reach_SE2_Box_EnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": rsl_rl_reach_se2_box_ppo_cfg.Go2_Reach_SE2_Box_PPORunnerCfg,
    },
)
