# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from rsl_rl.modules.actor_critic import ActorCritic as ActorCriticBase
from rsl_rl.utils import resolve_nn_activation
from rsl_rl_point_clouds.modules.point_cloud.point_conv_encoder import PointConvEncoder
from torch.distributions import Normal


class MLP(nn.Module):
    def __init__(self, num_input, num_output, hidden_dims=[256, 256, 256], activation="elu"):
        super().__init__()

        layers = []
        layers.append(nn.Linear(num_input, hidden_dims[0]))
        layers.append(activation)
        for layer_index in range(len(hidden_dims)):
            if layer_index == len(hidden_dims) - 1:
                layers.append(nn.Linear(hidden_dims[layer_index], num_output))
            else:
                layers.append(nn.Linear(hidden_dims[layer_index], hidden_dims[layer_index + 1]))
                layers.append(activation)
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ActorCritic(ActorCriticBase):
    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        point_cloud_shape,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        point_cloud_downsample_factor=4,
        point_cloud_downsample_steps=2,
        point_cloud_encoded_dim=256,
        learnable_std=True,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        nn.Module.__init__(self)
        activation = resolve_nn_activation(activation)

        self.point_cloud_encoder = PointConvEncoder(
            downsample_factor=point_cloud_downsample_factor, downsample_steps=point_cloud_downsample_steps
        )

        mlp_input_dim_a = num_actor_obs[0] + point_cloud_encoded_dim
        mlp_input_dim_c = num_critic_obs[0] + point_cloud_encoded_dim

        # Policy
        self.actor = MLP(mlp_input_dim_a, num_actions, actor_hidden_dims, activation)

        # Value function
        self.critic = MLP(mlp_input_dim_c, 1, critic_hidden_dims, activation)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")
        print(f"Point Cloud Encoder: {self.point_cloud_encoder}")

        # Action noise
        if learnable_std:
            self.noise_std_type = noise_std_type
            if self.noise_std_type == "scalar":
                self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            elif self.noise_std_type == "log":
                self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
            self.noise_std_type = "scalar"
            self.std = init_noise_std * torch.ones(num_actions)

        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    def _concat_observations(self, observations: torch.Tensor, point_cloud_encoded: torch.Tensor):
        return torch.cat((observations, point_cloud_encoded), dim=-1)

    def act(
        self,
        observations: torch.Tensor,
        point_cloud: torch.Tensor,
        point_cloud_encoded: torch.Tensor | None = None,
        **kwargs,
    ):
        if point_cloud_encoded is None:
            self.point_cloud_encoded = self.point_cloud_encoder(point_cloud)
        else:
            self.point_cloud_encoded = point_cloud_encoded
        return super().act(self._concat_observations(observations, self.point_cloud_encoded))

    def act_inference(
        self, observations: torch.Tensor, point_cloud: torch.Tensor, point_cloud_encoded: torch.Tensor | None = None
    ):
        if point_cloud_encoded is None:
            self.point_cloud_encoded = self.point_cloud_encoder(point_cloud)
        else:
            self.point_cloud_encoded = point_cloud_encoded
        return super().act_inference(self._concat_observations(observations, self.point_cloud_encoded))

    def evaluate(
        self,
        critic_observations: torch.Tensor,
        point_cloud: torch.Tensor,
        point_cloud_encoded: torch.Tensor | None = None,
        **kwargs,
    ):
        if point_cloud_encoded is None:
            self.point_cloud_encoded = self.point_cloud_encoder(point_cloud)
        else:
            self.point_cloud_encoded = point_cloud_encoded
        return super().evaluate(self._concat_observations(critic_observations, self.point_cloud_encoded))
