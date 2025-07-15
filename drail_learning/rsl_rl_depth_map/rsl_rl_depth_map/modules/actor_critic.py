from __future__ import annotations

import torch
import torch.nn as nn
from rsl_rl.modules.actor_critic import ActorCritic as ActorCriticBase
from rsl_rl.utils import resolve_nn_activation
from rsl_rl_depth_map.modules.depth_map.depth_map_encoder import DepthMapEncoder
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
        depth_map_shape,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        depth_map_encoded_dim=256,
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

        # Depth map encoder
        self.depth_map_encoder = DepthMapEncoder(in_channels=depth_map_shape[0], encoded_dim=depth_map_encoded_dim)

        mlp_input_dim_a = num_actor_obs[0] + depth_map_encoded_dim
        mlp_input_dim_c = num_critic_obs[0] + depth_map_encoded_dim

        # Policy
        self.actor = MLP(mlp_input_dim_a, num_actions, actor_hidden_dims, activation)

        # Value function
        self.critic = MLP(mlp_input_dim_c, 1, critic_hidden_dims, activation)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")
        print(f"Depth Map Encoder: {self.depth_map_encoder}")

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

    def _concat_observations(self, observations: torch.Tensor, depth_map_encoded: torch.Tensor):
        return torch.cat((observations, depth_map_encoded), dim=-1)

    def act(self, observations: torch.Tensor, depth_map: torch.Tensor, **kwargs):
        depth_map_encoded = self.depth_map_encoder(depth_map)
        return super().act(self._concat_observations(observations, depth_map_encoded))

    def act_inference(self, observations: torch.Tensor, depth_map: torch.Tensor):
        depth_map_encoded = self.depth_map_encoder(depth_map)
        return super().act_inference(self._concat_observations(observations, depth_map_encoded))

    def evaluate(self, critic_observations: torch.Tensor, depth_map: torch.Tensor, **kwargs):
        depth_map_encoded = self.depth_map_encoder(depth_map)
        return super().evaluate(self._concat_observations(critic_observations, depth_map_encoded))
