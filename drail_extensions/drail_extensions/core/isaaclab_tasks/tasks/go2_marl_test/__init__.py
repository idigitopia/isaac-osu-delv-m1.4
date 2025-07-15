import gymnasium as gym

from . import (
    marl_decentralized_multi_agent_test_env_cfg,
    marl_decentralized_single_agent_env_test_env_cfg,
    rsl_rl_marl_decentralized_test_ppo_cfg,
)

# Decentralized MARL (single agent)
gym.register(
    id="Go2-MARL-Decentralized-Single-Agent-Test-v0",
    entry_point="drail_extensions.core.envs:ManagerBasedMARLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            marl_decentralized_single_agent_env_test_env_cfg.Go2_MARL_Decentralized_Single_Agent_Test_EnvCfg
        ),
        "rsl_rl_cfg_entry_point": rsl_rl_marl_decentralized_test_ppo_cfg.Go2_MARL_Decentralized_PPO_Runner_Cfg,
    },
)

gym.register(
    id="Go2-MARL-Decentralized-Single-Agent-Test-Play-v0",
    entry_point="drail_extensions.core.envs:ManagerBasedMARLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            marl_decentralized_single_agent_env_test_env_cfg.Go2_MARL_Decentralized_Single_Agent_Test_EnvCfg_PLAY
        ),
        "rsl_rl_cfg_entry_point": rsl_rl_marl_decentralized_test_ppo_cfg.Go2_MARL_Decentralized_PPO_Runner_Cfg,
    },
)


# Decentralized MARL (multi-agent)
gym.register(
    id="Go2-MARL-Decentralized-Multi-Agent-Test-v0",
    entry_point="drail_extensions.core.envs:ManagerBasedMARLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            marl_decentralized_multi_agent_test_env_cfg.Go2_MARL_Decentralized_Multi_Agent_Test_EnvCfg
        ),
        "rsl_rl_cfg_entry_point": rsl_rl_marl_decentralized_test_ppo_cfg.Go2_MARL_Decentralized_PPO_Runner_Cfg,
    },
)

gym.register(
    id="Go2-MARL-Decentralized-Multi-Agent-Test-Play-v0",
    entry_point="drail_extensions.core.envs:ManagerBasedMARLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            marl_decentralized_multi_agent_test_env_cfg.Go2_MARL_Decentralized_Multi_Agent_Test_EnvCfg_PLAY
        ),
        "rsl_rl_cfg_entry_point": rsl_rl_marl_decentralized_test_ppo_cfg.Go2_MARL_Decentralized_PPO_Runner_Cfg,
    },
)
