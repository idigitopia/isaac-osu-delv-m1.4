# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from rsl_rl.modules import ActorCritic
from rsl_rl_drail.modules import ScaleLayer
from torch.distributions import Normal


# NOTE: Using different name from `ActorCritic` is a bit messy, but it is necessary to avoid conflicts
# with the base rsl_rl ActorCritic. This will allow for either to be used.
class DRAILActorCritic(ActorCritic):
    """
    Same as base rsl_rl ActorCritic but will use `getattr` for activation functions instead of internal
    `resolve_nn_activation` function. This way can use any `torch.nn` activation function without
    any modification.
    """

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        if "learnable_std" in kwargs and not kwargs["learnable_std"]:
            self.learnable_std = kwargs.pop("learnable_std")
        else:
            self.learnable_std = True
        if "bound_scale" in kwargs and kwargs["bound_scale"] is not None:
            self.bound_scale = kwargs.pop("bound_scale")
        else:
            self.bound_scale = None
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        nn.Module.__init__(self)
        try:
            activation = getattr(nn, activation)()
        except AttributeError:
            raise AttributeError(f"Activation function '{activation}' not found in torch.nn")

        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs
        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
                if self.bound_scale:
                    actor_layers.append(nn.Tanh())
                    actor_layers.append(ScaleLayer(self.bound_scale))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.noise_std_type = noise_std_type

        if self.noise_std_type not in {"scalar", "log"}:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        std_val = init_noise_std * torch.ones(num_actions)
        std_val = torch.log(std_val) if self.noise_std_type == "log" else std_val

        param = nn.Parameter(std_val) if self.learnable_std else std_val
        attr_name = "log_std" if self.noise_std_type == "log" else "std"

        if self.learnable_std:
            setattr(self, attr_name, param)
        else:
            self.register_buffer(attr_name, param)

        # Action distribution (populated in update_distribution)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

    def update_distribution(self, observations):
        # compute mean
        mean = self.actor(observations)
        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        std = torch.clamp(std, min=1e-8)
        # create distribution
        self.distribution = Normal(mean, std)

    def get_actions_log_prob(self, actions):
        log_prob = self.distribution.log_prob(actions).sum(dim=-1)
        if self.bound_scale:
            # SAC, Appendix C, https://arxiv.org/pdf/1801.01290.pdf
            log_prob -= torch.log(self.bound_scale * (1 - torch.tanh(actions).pow(2)) + 1e-6).sum(dim=-1)

        return log_prob
