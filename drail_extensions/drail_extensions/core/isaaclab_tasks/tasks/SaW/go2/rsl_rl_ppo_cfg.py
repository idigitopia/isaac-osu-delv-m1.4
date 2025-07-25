# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional

from isaaclab.utils import configclass

from drail_extensions.core.isaaclab_rsl_rl.utils import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class Go2FlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 50
    experiment_name = "unitree_go2_flat"
    empirical_normalization = False
    init_at_random_ep_len = True  # Whether to initialize episodes at random lengths for better exploration

    policy = RslRlPpoActorCriticCfg(
        # This will be used to determine the policy class
        class_name="rsl_rl.modules.ActorCritic",
        init_noise_std=1.0,
        actor_hidden_dims=[128, 128, 128],
        critic_hidden_dims=[128, 128, 128],
        activation="elu",
        learnable_std=True,
    )

    algorithm = RslRlPpoAlgorithmCfg(
        # This will be used to determine the algorithm class
        class_name="rsl_rl.algorithms.PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

    # This will be used to determine the runner class
    class_name = "rsl_rl.runners.OnPolicyRunner"

    # extra things
    wandb_checkpoint: Optional[str] = ""
    raw_checkpoint: Optional[str] = ""


@configclass
class Go2_Velocity_SpotReward_RunnerCfg(Go2FlatPPORunnerCfg):
    experiment_name = "go2_velocity_spot_reward"
