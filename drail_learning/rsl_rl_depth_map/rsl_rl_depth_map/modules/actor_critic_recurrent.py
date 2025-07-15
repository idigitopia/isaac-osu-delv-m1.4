from __future__ import annotations

import rsl_rl  # noqa: F401
import torch
import torch.nn as nn
from rsl_rl.modules.actor_critic_recurrent import Memory
from rsl_rl.utils import resolve_nn_activation
from rsl_rl_depth_map.modules.actor_critic import MLP, ActorCritic
from rsl_rl_depth_map.modules.depth_map.depth_map_encoder import DepthMapEncoder
from torch.distributions import Normal


class ActorCriticRecurrent(ActorCritic):
    is_recurrent = True

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        depth_map_shape,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_size=256,
        rnn_num_layers=1,
        depth_map_encoded_dim=256,
        init_noise_std=1.0,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticRecurrent.__init__ got unexpected arguments, which will be ignored: " + str(kwargs.keys()),
            )
        nn.Module.__init__(self)
        activation = resolve_nn_activation(activation)

        # Depth map encoder
        self.depth_map_encoder = DepthMapEncoder(in_channels=depth_map_shape[0], encoded_dim=depth_map_encoded_dim)

        rnn_input_dim_a = self._get_mlp_input_dim(num_actor_obs, depth_map_shape, depth_map_encoded_dim)
        rnn_input_dim_c = self._get_mlp_input_dim(num_critic_obs, depth_map_shape, depth_map_encoded_dim)

        # RNN Memory
        self.memory_a = Memory(rnn_input_dim_a, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)
        self.memory_c = Memory(rnn_input_dim_c, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)

        # Policy
        self.actor = MLP(rnn_hidden_size, num_actions, actor_hidden_dims, activation)

        # Value function
        self.critic = MLP(rnn_hidden_size, 1, critic_hidden_dims, activation)

        print(f"Depth Map Encoder: {self.depth_map_encoder}")
        print(f"Actor RNN Memory: {self.memory_a}")
        print(f"Critic RNN Memory: {self.memory_c}")
        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

    def reset(self, dones=None):
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)

    def _forward_rnn(
        self,
        rnn_model: nn.Module,
        observations: torch.Tensor,
        depth_map: torch.Tensor,
        masks: torch.Tensor = None,
        hidden_states: list[torch.Tensor] = None,
    ):
        depth_map_encoded = self.depth_map_encoder(depth_map)
        rnn_input = self._concat_observations(observations, depth_map_encoded)
        return rnn_model(rnn_input, masks, hidden_states)

    def act(
        self,
        observations: torch.Tensor,
        depth_map: torch.Tensor,
        masks: torch.Tensor = None,
        hidden_states: list[torch.Tensor] = None,
    ):
        rnn_output = self._forward_rnn(self.memory_a, observations, depth_map, masks, hidden_states)
        return super(ActorCritic, self).act(rnn_output.squeeze(0))

    def act_inference(self, observations, depth_map: torch.Tensor):
        rnn_output = self._forward_rnn(self.memory_a, observations, depth_map)
        return super(ActorCritic, self).act_inference(rnn_output.squeeze(0))

    def evaluate(self, critic_observations, depth_map: torch.Tensor, masks=None, hidden_states=None):
        rnn_output = self._forward_rnn(self.memory_c, critic_observations, depth_map, masks, hidden_states)
        return super(ActorCritic, self).evaluate(rnn_output.squeeze(0))

    def get_hidden_states(self):
        return self.memory_a.hidden_states, self.memory_c.hidden_states
