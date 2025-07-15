# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

import random

from drail_extensions.research_ashton.agents import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class Go2PoseWrapperPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = random.randint(0, 100000)
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = max_iterations // 10
    experiment_name = "go2_pose_wrapper"
    empirical_normalization = False

    # Policy parameters
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        learnable_std=True,
    )
    # PPO Algorithm parameters
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
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
        lr_scheduler=False,
    )

    clip_actions = 20.0

    class_name = "rsl_rl_ashton.runners.OnPolicyRunner"

    run_name = "n_spot"  # Name of the run
    logger = "wandb"
    wandb_project = "isaaclabdrail"
    wandb_group = "go2_pose_wrapper"
    wandb_entity = "ashton-drail"
