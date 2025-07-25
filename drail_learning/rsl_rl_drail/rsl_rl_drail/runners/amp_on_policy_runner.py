# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import os
import pickle
import statistics
import time
from collections import deque

import rsl_rl_drail  # noqa: F401, E402
import rsl_rl_drail.runners as drail_runners
import torch
from rsl_rl.env import VecEnv
from rsl_rl.modules import EmpiricalNormalization
from rsl_rl.utils import store_code_state
from rsl_rl_drail.algorithms import AMPPPO


class AMPOnPolicyRunner(drail_runners.OnPolicyRunner):
    """On-policy runner for training and evaluation."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        self.cfg = train_cfg
        self.discriminator_cfg = train_cfg["discriminator"]
        discriminator_class = eval(self.discriminator_cfg.pop("class_name"))  # ActorCritic

        # resolve dimensions of observations
        obs, extras = env.get_observations()
        self.num_amp_obs = extras["observations"]["amp"].shape[1] if "amp" in extras["observations"] else obs.shape[1]

        full_class = discriminator_class(
            num_actor_obs=self.num_amp_obs,  # discriminator input observation
            num_critic_obs=1,  # not used
            num_actions=1,  # discriminator output logits
            **self.discriminator_cfg,
        )
        # self.is_discrim_recurrent = full_class.is_recurrent
        self.discriminator = full_class.actor.to(device)
        self.discriminator.is_recurrent = full_class.is_recurrent
        # init algorithm
        self._load_demo_data(env)

        # init algorithm with super init
        super().__init__(env, train_cfg, log_dir, device)

        # Normalize amp observations
        self.amp_empirical_normalization = getattr(self.cfg, "amp_empirical_normalization", False)

        if self.amp_empirical_normalization:
            self.amp_obs_normalizer = EmpiricalNormalization(shape=[self.num_amp_obs], until=1.0e8).to(self.device)
        else:
            self.amp_obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization

    def _load_demo_data(self, env: VecEnv):
        def process_demo_file(file_path):
            with open(file_path, "rb") as f:
                traj = pickle.load(f)
                state_keys = ["q", "dq"]
                history_length = env.cfg.observations.amp.history_length

                data_list = []
                for state_key in state_keys:
                    data = torch.tensor(traj[state_key])
                    data = data.unfold(0, history_length, 1).transpose(1, 2).reshape(-1, history_length * data.size(-1))
                    data_list.append(data)
                return torch.cat(data_list, dim=-1)

        if isinstance(self.cfg["obs_demo_path"], list):
            self.traj_data = torch.cat([process_demo_file(f) for f in self.cfg["obs_demo_path"]], dim=0)
        else:
            self.traj_data = process_demo_file(self.cfg["obs_demo_path"])

    def _init_algorithm_and_storage(self, policy, num_obs, num_privileged_obs):
        # TODO: Should we even do the eval here, or just hardcode to "AMPPPO"?
        alg_class = eval(self.alg_cfg.pop("class_name"))  # AMPPPO
        self.alg: AMPPPO = alg_class(
            policy,
            discriminator=self.discriminator,
            obs_demo=self.traj_data,
            discriminator_l2_reg=self.cfg["discriminator_l2_reg"],
            discriminator_grad_penalty=self.cfg["discriminator_grad_penalty"],
            device=self.device,
            **self.alg_cfg,
        )
        self.alg.init_storage(
            self.training_type,
            self.env.num_envs,
            self.num_steps_per_env,
            [num_obs],
            [num_privileged_obs],
            [self.num_amp_obs],
            [self.env.num_actions],
        )

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):

        # Get the style reward term defined in reward cfg
        self.style_reward_term = self.env.unwrapped.reward_manager.cfg.style_reward

        # Set the style evaluator function
        if self.style_reward_term.weight > 0:

            def style_evaluator(amp_obs):
                self.alg.discriminator.eval()
                return self.alg.discriminator(amp_obs)

            self.style_reward_term.func.compute_logits = style_evaluator

        self.alg.train_discriminator_flag = self.style_reward_term.weight > 0

        # initialize writer
        if self.log_dir is not None and self.writer is None and not self.disable_logs:
            # Launch either Tensorboard or Neptune & Tensorboard summary writer(s), default: Tensorboard.
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from rsl_rl_drail.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise ValueError("Logger type not found. Please choose 'neptune', 'wandb' or 'tensorboard'.")

        # randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # start learning
        obs, extras = self.env.get_observations()
        # critic_obs = extras["observations"].get("critic", obs)
        privileged_obs = extras["observations"].get(self.privileged_obs_type, obs)
        amp_obs = extras["observations"]["amp"]
        # obs, critic_obs, amp_obs = obs.to(self.device), critic_obs.to(self.device), amp_obs.to(self.device)
        obs, privileged_obs, amp_obs = obs.to(self.device), privileged_obs.to(self.device), amp_obs.to(self.device)
        amp_obs = self.amp_obs_normalizer(amp_obs)
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

        # Ensure all parameters are in-synced
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        # Keep track of the maximum mean episode reward
        max_reward = -math.inf
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):

                    # Sample actions from policy
                    actions = self.alg.act(obs, privileged_obs, amp_obs)
                    # Step environment
                    obs, rewards, dones, infos = self.env.step(actions.to(self.env.device))

                    # Move to the agent device
                    obs, rewards, dones = obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    amp_obs = self.amp_obs_normalizer(infos["observations"]["amp"].to(self.device))

                    # Normalize observations
                    obs = self.obs_normalizer(obs)
                    if self.privileged_obs_type is not None:
                        privileged_obs = self.privileged_obs_normalizer(
                            infos["observations"][self.privileged_obs_type].to(self.device)
                        )
                    else:
                        privileged_obs = obs

                    # Process env step and store in buffer
                    self.alg.process_env_step(rewards, dones, infos)

                    # Intrinsic rewards (extracted here only for logging)!
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.alg.rnd else None

                    if self.log_dir is not None:
                        # Book keeping
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
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
                start = stop

                # compute returns
                if self.training_type == "rl":
                    self.alg.compute_returns(privileged_obs)

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

    def train_mode(self):
        super().train_mode()
        self.alg.discriminator.train()

    def eval_mode(self):
        super().eval_mode()
        self.alg.discriminator.eval()

    def save(self, path: str, infos=None, extras=None):
        disc_info = {
            "discriminator_state_dict": self.alg.discriminator.state_dict(),
            "discriminator_optimizer_state_dict": self.alg.optimizer_discriminator.state_dict(),
        }
        extras = {} if extras is None else extras
        extras.update(disc_info)

        super().save(path, infos, extras=extras)

    def load(self, path: str, load_optimizer: bool = True):
        try:
            loaded_dict = torch.load(path, weights_only=False)
            self.alg.discriminator.load_state_dict(loaded_dict["discriminator_state_dict"])
            if load_optimizer:
                self.alg.optimizer_discriminator.load_state_dict(loaded_dict["discriminator_optimizer_state_dict"])
        except Exception as e:
            print(f"Error loading discriminator: {e}")
        return super().load(path, load_optimizer)
