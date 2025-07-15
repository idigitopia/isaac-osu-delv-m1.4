# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv

from .vecenv_wrapper import RslRlVecEnvWrapper


class VecEnvDemarlWrapper(RslRlVecEnvWrapper):
    def __init__(self, env: Any):
        """Isaac Lab environment wrapper for multi-agent implementation

        :param env: The environment to wrap
        :type env: Any supported Isaac Lab environment

        Note:
            The wrapper calls :meth:`reset` at the start since the RSL-RL runner does not call reset.

        Args:
            env: The environment to wrap around.

        Raises:
            ValueError: When the environment is not an instance of :class:`ManagerBasedRLEnv` or :class:`DirectRLEnv`.
        """
        # check that input is valid
        if not isinstance(env.unwrapped, ManagerBasedRLEnv):
            raise ValueError(
                "The environment must be inherited from ManagerBasedRLEnv or DirectRLEnv. Environment type:"
                f" {type(env)}"
            )
        # initialize the wrapper
        self.env = env
        # store information required by wrapper
        self.num_envs = self.unwrapped.num_envs
        self.device = self.unwrapped.device
        self.max_episode_length = self.unwrapped.max_episode_length

        self._configure_gym_env_spaces()
        self._configure_network_shapes()
        self.env.reset()

    @property
    def num_actions(self) -> int:
        return self._num_actions

    @property
    def num_obs(self) -> int:
        return self._num_obs

    @property
    def num_privileged_obs(self) -> int:
        return self._num_privileged_obs

    def _configure_network_shapes(self):
        if hasattr(self.unwrapped, "action_manager"):
            self._num_actions = self.unwrapped.action_manager.total_action_dim // self.num_agents
        else:
            self._num_actions = gym.spaces.flatdim(self.unwrapped.single_action_space)
        if hasattr(self.unwrapped, "observation_manager"):
            self._num_obs = self.unwrapped.observation_manager.group_obs_dim["policy"][0]
        else:
            self._num_obs = gym.spaces.flatdim(self.unwrapped.single_observation_space["policy"])
        # -- privileged observations
        if (
            hasattr(self.unwrapped, "observation_manager")
            and "critic" in self.unwrapped.observation_manager.group_obs_dim
        ):
            self._num_privileged_obs = self.unwrapped.observation_manager.group_obs_dim["critic"][0]
        elif hasattr(self.unwrapped, "num_states") and "critic" in self.unwrapped.single_observation_space:
            self._num_privileged_obs = gym.spaces.flatdim(self.unwrapped.single_observation_space["critic"])
        else:
            self._num_privileged_obs = 0

    def _parse_term_agent_id(self, key: str) -> tuple[str, int]:
        if not hasattr(self, "_agent_id_pattern"):
            self._agent_id_pattern = re.compile(r"^(.*?)(?:_(\d+))?$")
        match = self._agent_id_pattern.match(key)
        if not match:
            raise ValueError(f"Invalid key: {key}. Expected format for term name: <base_key>[_<agent_id>]")
        term_name, agent_id = match.groups()
        agent_id = int(agent_id) if agent_id is not None else 0
        return term_name, agent_id

    def _configure_observation_spaces(self):
        single_observation_space = self.env.unwrapped.single_observation_space

        policy_obs_space = single_observation_space["policy"]
        critic_obs_space = (
            single_observation_space["critic"]
            if "critic" in single_observation_space.keys()  # noqa: SIM118
            else policy_obs_space
        )

        for agent_id in range(1, self.num_agents):
            policy_obs_key, critic_obs_key = f"policy_{agent_id}", f"critic_{agent_id}"
            if policy_obs_key not in single_observation_space.keys():
                break
            assert policy_obs_space == single_observation_space[policy_obs_key], (
                f"Agent {agent_id} has policy observation space {single_observation_space[policy_obs_key]} in key"
                f" {policy_obs_key} but expected {policy_obs_space}"
            )
            if critic_obs_key in single_observation_space.keys():  # noqa: SIM118
                assert critic_obs_space == single_observation_space[critic_obs_key], (
                    f"Agent {agent_id} has critic observation space {single_observation_space[critic_obs_key]} in key"
                    f" {critic_obs_key} but expected {critic_obs_space}"
                )

        self._observation_spaces = gym.spaces.Dict(
            {
                "policy": gym.vector.utils.batch_space(policy_obs_space, self.num_envs * self.num_agents),
                "critic": gym.vector.utils.batch_space(critic_obs_space, self.num_envs * self.num_agents),
            }
        )

    def _configure_action_spaces(self):
        action_spaces = [0] * self.num_agents
        for name, space in zip(
            self.env.unwrapped.action_manager._term_names, self.env.unwrapped.action_manager.action_term_dim
        ):
            _, agent_id = self._parse_term_agent_id(name)
            action_spaces[agent_id] += space
        assert all(
            action_spaces[0] == action_spaces[i] for i in range(1, self.num_agents)
        ), "All agents must have the same action space"
        self._action_spaces = gym.vector.utils.batch_space(
            gym.spaces.Box(low=-np.inf, high=np.inf, shape=(space,)), self.num_envs * self.num_agents
        )

    def _configure_gym_env_spaces(self):
        self._configure_observation_spaces()
        self._configure_action_spaces()

    @property
    def observation_space(self) -> gym.Space:
        return self._observation_spaces

    @property
    def action_space(self) -> gym.Space:
        return self._action_spaces

    @property
    def num_agents(self) -> int:
        return self.env.unwrapped.cfg.num_agents

    def get_observations(self, obs_dict: dict[str, torch.Tensor] | None = None) -> tuple[torch.Tensor, dict]:
        if obs_dict is None:
            obs_dict = self.unwrapped.observation_manager.compute()
        observations = defaultdict(dict)

        policy_obs = []
        critic_obs = []

        for agent_id in range(self.num_agents):
            policy_obs_key = f"policy_{agent_id}" if agent_id > 0 else "policy"
            critic_obs_key = f"critic_{agent_id}" if agent_id > 0 else "critic"
            policy_obs.append(obs_dict[policy_obs_key])
            critic_obs.append(obs_dict.get(critic_obs_key, obs_dict[policy_obs_key]))

        policy_obs = torch.cat(policy_obs)
        critic_obs = torch.cat(critic_obs)

        for key, value in obs_dict.items():
            term_name, agent_id = self._parse_term_agent_id(key)
            observations[f"agent_{agent_id}"][term_name] = value

        return policy_obs, {"observations": {"policy": policy_obs, "critic": critic_obs}}

    def _get_rewards(self, reward_dict: dict[str, dict[str, torch.Tensor]]) -> tuple[torch.Tensor, dict]:
        rewards = []
        for agent_id in range(self.num_agents):
            reward_key = f"reward_{agent_id}" if agent_id > 0 else "reward"
            rewards.append(reward_dict[reward_key])
        rewards = torch.cat(rewards)
        return rewards

    def _get_dones(
        self, terminated: dict[str, torch.Tensor], truncated: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict]:
        # If any agent has terminated in an environment, set all agents to terminated
        # Note: "truncated" will be used to bootstrap the value function for agents that terminated
        # because of other agents
        return self.env.unwrapped.termination_manager.dones.repeat(self.num_agents)

    def _get_time_outs(
        self, terminated: dict[str, torch.Tensor], truncated: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict]:
        terminated_tensor = []
        truncated_tensor = []
        for agent_id in range(self.num_agents):
            done_key = f"done_{agent_id}" if agent_id > 0 else "done"
            terminated_tensor.append(terminated[done_key])
            truncated_tensor.append(truncated[done_key])

        terminated_tensor = torch.stack(terminated_tensor)
        truncated_tensor = torch.stack(truncated_tensor)

        # Check if any agent has terminated in an environment
        any_terminated = terminated_tensor.any(0)

        # If agent has not terminated but any other agent has terminated in an environment, set truncated to true
        truncated_tensor |= any_terminated & ~terminated_tensor

        return truncated_tensor.view(-1)

    def reset(self) -> tuple[torch.Tensor, dict]:
        """Reset the environment

        :return: Observation, info
        :rtype: torch.Tensor and any other info
        """
        obs_dict, _ = self.env.reset()

        return self.get_observations(obs_dict)

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Perform a step in the environment

        :param actions: The actions to perform
        :type actions: dictionary of torch.Tensor

        :return: Observation, reward, terminated, truncated, info
        :rtype: tuple of dictionaries torch.Tensor and any other info
        """
        obs_dict, reward_dict, terminated, truncated, extras = self.env.step(
            actions.view(self.num_agents, self.num_envs, -1).transpose(0, 1).reshape(self.num_envs, -1)
        )

        policy_obs, obs_dict = self.get_observations(obs_dict)
        extras["observations"] = obs_dict["observations"]
        rewards = self._get_rewards(reward_dict)
        dones = self._get_dones(terminated, truncated)
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = self._get_time_outs(terminated, truncated)
        else:
            extras["time_outs"] = {}

        return policy_obs, rewards, dones, extras


class VecEnvDemarlCentralizedCriticWrapper(VecEnvDemarlWrapper):
    def __init__(self, env: Any):
        super().__init__(env)

    def _configure_observation_spaces(self):
        super()._configure_observation_spaces()

        critic_obs_shape = list(self.observation_space["critic"].shape)
        critic_obs_shape[1] *= self.num_agents
        self._observation_spaces["critic"] = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=critic_obs_shape,
        )

    def get_observations(self, obs_dict: dict[str, torch.Tensor] | None = None) -> tuple[torch.Tensor, dict]:
        if obs_dict is None:
            obs_dict = self.unwrapped.observation_manager.compute()
        observations = defaultdict(dict)

        policy_obs = []
        critic_obs = []

        for agent_id in range(self.num_agents):
            policy_obs_key = f"policy_{agent_id}" if agent_id > 0 else "policy"
            critic_obs_key = f"critic_{agent_id}" if agent_id > 0 else "critic"
            policy_obs.append(obs_dict[policy_obs_key])
            critic_obs.append(obs_dict.get(critic_obs_key, obs_dict[policy_obs_key]))

        policy_obs = torch.cat(policy_obs)
        # Centralized critic observation, repeat each agent's critic observation for all other agents
        critic_obs = torch.stack(critic_obs, dim=1).view(self.num_envs, -1).repeat(self.num_agents, 1)

        for key, value in obs_dict.items():
            term_name, agent_id = self._parse_term_agent_id(key)
            observations[f"agent_{agent_id}"][term_name] = value

        return policy_obs, {"observations": {"policy": policy_obs, "critic": critic_obs}}
