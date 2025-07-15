# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import os
import statistics
import time
import torch
from collections import deque

import rsl_rl
from rsl_rl.env import VecEnv
from rsl_rl.modules.normalizer import EmpiricalNormalization
from rsl_rl.runners.on_policy_runner import OnPolicyRunner as OnPolicyRunnerBase
from rsl_rl.utils import store_code_state

import rsl_rl_depth_map
from rsl_rl_depth_map.algorithms import PPO
from rsl_rl_depth_map.modules import ActorCritic, ActorCriticRecurrent


class OnPolicyRunner(OnPolicyRunnerBase):
    """On-policy runner for training and evaluation."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        # resolve dimensions of observations
        obs, extras = self.env.get_observations()
        self.num_obs = obs.shape[1:]
        self.num_critic_obs = (
            extras["observations"]["critic"].shape[1:] if "critic" in extras["observations"] else self.num_obs
        )
        self.num_depth_map_obs = extras["observations"]["depth_map"].shape[1:]
        actor_critic_class = eval(self.policy_cfg.get("class_name"))  # ActorCritic
        self.actor_critic: ActorCritic | ActorCriticRecurrent = actor_critic_class(
            self.num_obs, self.num_critic_obs, self.num_depth_map_obs, self.env.num_actions, **self.policy_cfg
        ).to(self.device)

        # resolve dimension of rnd gated state
        if "rnd_cfg" in self.alg_cfg and self.alg_cfg["rnd_cfg"] is not None:
            # check if rnd gated state is present
            rnd_state = extras["observations"].get("rnd_state")
            if rnd_state is None:
                raise ValueError("Observations for they key 'rnd_state' not found in infos['observations'].")
            # get dimension of rnd gated state
            num_rnd_state = rnd_state.shape[1]
            # add rnd gated state to config
            self.alg_cfg["rnd_cfg"]["num_state"] = num_rnd_state
            # scale down the rnd weight with timestep (similar to how rewards are scaled down in legged_gym envs)
            self.alg_cfg["rnd_cfg"]["weight"] *= env.dt

        # if using symmetry then pass the environment config object
        if "symmetry_cfg" in self.alg_cfg and self.alg_cfg["symmetry_cfg"] is not None:
            # this is used by the symmetry function for handling different observation terms
            self.alg_cfg["symmetry_cfg"]["_env"] = env

        # store training configuration
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.empirical_normalization = self.cfg["empirical_normalization"]

        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=self.num_obs, until=1.0e8).to(self.device)
            self.critic_obs_normalizer = EmpiricalNormalization(shape=[self.num_critic_obs], until=1.0e8).to(
                self.device
            )
        else:
            self.obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization
            self.critic_obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__, rsl_rl_depth_map.__file__]

        # init algorithm
        self.init_algorithm_and_storage()

    def init_algorithm_and_storage(self):
        alg_class = eval(self.alg_cfg.get("class_name"))  # PPO
        self.alg: PPO = alg_class(self.actor_critic, device=self.device, **self.alg_cfg)
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            self.num_obs,
            self.num_critic_obs,
            self.num_depth_map_obs,
            [self.env.num_actions],
        )

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        self.init_logger()
        # randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # start learning
        obs, extras = self.env.get_observations()
        critic_obs = extras["observations"].get("critic", obs)
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.train_mode()  # switch to train mode (for dropout for example)

        # Book keeping
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        # create buffers for logging extrinsic and intrinsic rewards
        if self.alg.rnd:
            erewbuffer = deque(maxlen=100)
            irewbuffer = deque(maxlen=100)
            cur_ereward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            cur_ireward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations

        # Keep track of the maximum mean episode reward
        max_reward = -math.inf

        for it in range(start_iter, tot_iter):
            start = time.time()
            # Rollout
            with torch.no_grad():
                for _ in range(self.num_steps_per_env):
                    # Sample actions from policy
                    depth_map_obs = extras["observations"]["depth_map"]
                    actions = self.alg.act(obs, critic_obs, depth_map_obs)
                    # Step environment
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))

                    # Move to the agent device
                    obs, rewards, dones = obs.to(self.device), rewards.to(self.device), dones.to(self.device)

                    # Normalize observations
                    obs = self.obs_normalizer(obs)
                    # Extract critic observations and normalize
                    if "critic" in extras["observations"]:
                        critic_obs = self.critic_obs_normalizer(extras["observations"]["critic"].to(self.device))
                    else:
                        critic_obs = obs

                    # Intrinsic rewards (extracted here only for logging)!
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.alg.rnd else None

                    # Process env step and store in buffer
                    self.alg.process_env_step(rewards, dones, extras)

                    if self.log_dir is not None:
                        # Book keeping
                        if "episode" in extras:
                            ep_infos.append(extras["episode"])
                        elif "log" in extras:
                            ep_infos.append(extras["log"])
                        # Update rewards
                        if self.alg.rnd:
                            cur_ereward_sum += rewards
                            cur_ireward_sum += intrinsic_rewards  # type: ignore
                            cur_reward_sum += rewards + intrinsic_rewards
                        else:
                            cur_reward_sum += rewards
                        # Update episode length
                        cur_episode_length += 1
                        # Clear data for completed episodes
                        # -- common
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                        # -- intrinsic and extrinsic rewards
                        if self.alg.rnd:
                            erewbuffer.extend(cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            irewbuffer.extend(cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            cur_ereward_sum[new_ids] = 0
                            cur_ireward_sum[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                depth_map_obs = extras["observations"]["depth_map"]
                self.alg.compute_returns(critic_obs, depth_map_obs)

            # Update policy
            # Note: we keep arguments here since locals() loads them
            loss_dict = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            # Logging info and save checkpoint
            if self.log_dir is not None:
                # Log information
                self.log(locals())
                # Save model
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
                self.save(os.path.join(self.log_dir, "model_latest.pt"))

                if len(rewbuffer) > 0 and (new_max_reward := statistics.mean(rewbuffer)) > max_reward:
                    max_reward = new_max_reward
                    self.save(os.path.join(self.log_dir, "model_best.pt"))

            # Clear episode infos
            ep_infos.clear()

            # Save code state
            if it == start_iter:
                # obtain all the diff files
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # if possible store them to wandb
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        # Save the final model after training
        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))
