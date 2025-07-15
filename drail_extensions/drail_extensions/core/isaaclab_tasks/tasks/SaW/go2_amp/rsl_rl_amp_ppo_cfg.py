# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Optional

from isaaclab.utils import configclass

from drail_extensions.core.isaaclab_rsl_rl.utils import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


class RslRlAMPOnPolicyRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Configuration of the runner for on-policy algorithms."""

    discriminator_l2_reg: float = MISSING
    discriminator_grad_penalty: float = MISSING
    obs_demo_path: str = MISSING
    amp_empirical_normalization: bool = False

    # Note: Only actor network is used for discriminator
    discriminator: RslRlPpoActorCriticCfg = MISSING


@configclass
class Go2AMPFlatPPORunnerCfg(RslRlAMPOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 50
    experiment_name = "unitree_go2_amp_flat"
    empirical_normalization = False
    amp_empirical_normalization = True

    # Discriminator Config
    discriminator_l2_reg = 0.01
    discriminator_grad_penalty = 5.0
    obs_demo_path = [
        "drail_extensions/drail_extensions/core/data/assets/go2_motions/WalkMotion.pk",
        "drail_extensions/drail_extensions/core/data/assets/go2_motions/SideMotion.pk",
        "drail_extensions/drail_extensions/core/data/assets/go2_motions/TurnMotion.pk",
        "drail_extensions/drail_extensions/core/data/assets/go2_motions/StandMotion.pk",
    ]

    # Policy Network Actor Critic Config
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[128, 128, 128],
        critic_hidden_dims=[128, 128, 128],
        activation="elu",
        learnable_std=True,
    )

    # Policy Network  Discriminator Config
    discriminator = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[1024, 512],
        critic_hidden_dims=[512, 256],
        activation="elu",
    )

    # PPO Algorithm Config
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="AMPPPO",
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

    # Important . This will be used to determine the runner class
    class_name = "rsl_rl.runners.AMPOnPolicyRunner"

    # extra things
    wandb_checkpoint: Optional[str] = ""
    raw_checkpoint: Optional[str] = ""
