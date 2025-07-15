# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Literal

import isaaclab_rl.rsl_rl.rl_cfg as base_rsl_rl_cfg
from isaaclab.utils import configclass

from .rnd_cfg import RslRlRndCfg
from .symmetry_cfg import RslRlSymmetryCfg

# RslRlPpoAlgorithmCfg, RslRlPpoActorCriticCfg, RslRlOnPolicyRunnerCfg


@configclass
class RslRlPpoActorCriticCfg(base_rsl_rl_cfg.RslRlPpoActorCriticCfg):
    """Configuration for the PPO actor-critic networks."""

    learnable_std: bool = True
    """Whether to make the std learnable. Default is True."""

    noise_std_type: Literal["scalar", "log"] = "scalar"
    """The type of noise to use for the std. Default is "scalar"."""


@configclass
class RslRlPpoAlgorithmCfg(base_rsl_rl_cfg.RslRlPpoAlgorithmCfg):
    """Configuration for the PPO algorithm."""

    symmetry_cfg: RslRlSymmetryCfg | None = None
    """The symmetry configuration. Default is None, in which case symmetry is not used."""

    rnd_cfg: RslRlRndCfg | None = None
    """The configuration for the Random Network Distillation (RND) module. Default is None,
    in which case RND is not used.
    """


@configclass
class RslRlOnPolicyRunnerCfg(base_rsl_rl_cfg.RslRlOnPolicyRunnerCfg):
    """Configuration of the runner for on-policy algorithms."""

    wandb_checkpoint: str = ""
    """The wandb checkpoint to load. Default is empty string."""

    raw_checkpoint: str = ""
    """The raw checkpoint to load. Default is empty string."""
