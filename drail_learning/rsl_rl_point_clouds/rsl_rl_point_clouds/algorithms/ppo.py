# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from rsl_rl.algorithms.ppo import PPO as PPOBase
from rsl_rl_point_clouds.modules import ActorCritic
from rsl_rl_point_clouds.storage import RolloutStorage


class PPO(PPOBase):
    def __init__(self, *args, **kwargs):
        self.point_cloud_encoder_learning_freq = kwargs.pop("point_cloud_encoder_learning_freq")
        super().__init__(*args, **kwargs)

    """Adds point cloud observations to the base PPO algorithm."""

    actor_critic: ActorCritic
    """The actor critic module."""

    def init_storage(
        self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, point_cloud_obs_shape, action_shape
    ):
        # create memory for RND as well :)
        if self.rnd:
            rnd_state_shape = [self.rnd.num_states]
        else:
            rnd_state_shape = None
        # create rollout storage
        self.storage = RolloutStorage(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            point_cloud_obs_shape,
            action_shape,
            rnd_state_shape,
            self.device,
        )

    def act(self, obs, critic_obs, point_cloud_obs):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs, point_cloud_obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs, point_cloud_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        self.transition.point_cloud_observations = point_cloud_obs
        return self.transition.actions

    def compute_returns(self, last_critic_obs, last_point_cloud_obs):
        # compute value for the last step
        last_values = self.actor_critic.evaluate(last_critic_obs, last_point_cloud_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):  # noqa: C901
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        # -- RND loss
        if self.rnd:
            mean_rnd_loss = 0
        else:
            mean_rnd_loss = None
        # -- Symmetry loss
        if self.symmetry:
            mean_symmetry_loss = 0
        else:
            mean_symmetry_loss = None

        # generator for mini batches
        if self.actor_critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        point_cloud_encoded_features_actor_cached = None
        point_cloud_encoded_features_critic_cached = None

        # iterate over batches
        for processed_mini_batch in generator:
            obs_batch = processed_mini_batch["obs_batch"]
            critic_obs_batch = processed_mini_batch["critic_obs_batch"]
            point_cloud_obs_batch = processed_mini_batch["point_cloud_obs_batch"]
            actions_batch = processed_mini_batch["actions_batch"]
            target_values_batch = processed_mini_batch["target_values_batch"]
            advantages_batch = processed_mini_batch["advantages_batch"]
            returns_batch = processed_mini_batch["returns_batch"]
            old_actions_log_prob_batch = processed_mini_batch["old_actions_log_prob_batch"]
            old_mu_batch = processed_mini_batch["old_mu_batch"]
            old_sigma_batch = processed_mini_batch["old_sigma_batch"]
            hid_states_batch = processed_mini_batch.get("hid_a"), processed_mini_batch.get("hid_c")
            masks_batch = processed_mini_batch.get("obs_mask_batch")
            rnd_state_batch = processed_mini_batch.get("rnd_state_batch")
            epoch = processed_mini_batch.get("epoch")
            batch_size = processed_mini_batch.get("batch_size")
            batch_idx = processed_mini_batch.get("batch_idx")

            # number of augmentations per sample
            # we start with 1 and increase it if we use symmetry augmentation
            num_aug = 1
            # original batch size
            original_batch_size = obs_batch.shape[0]

            # Perform symmetric augmentation
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                # augmentation using symmetry
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                # returned shape: [batch_size * num_aug, ...]
                obs_batch, point_cloud_obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch,
                    point_cloud_obs=point_cloud_obs_batch,
                    actions=actions_batch,
                    env=self.symmetry["_env"],
                    is_critic=False,
                )
                critic_obs_batch, _ = data_augmentation_func(
                    obs=critic_obs_batch, point_cloud_obs=None, actions=None, env=self.symmetry["_env"], is_critic=True
                )
                # compute number of augmentations per sample
                num_aug = int(obs_batch.shape[0] / original_batch_size)
                # repeat the rest of the batch
                # -- actor
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                # -- critic
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

            # Recompute actions log prob and entropy for current batch of transitions
            # Note: we need to do this because we updated the actor_critic with the new parameters
            # -- actor
            self.actor_critic.act(
                obs_batch,
                point_cloud_obs_batch,
                # If epoch % point_cloud_encoder_learning_freq == 0, then we compute the point cloud encoded features
                # using the point cloud encoder. Otherwise, we use the previously computed point cloud encoded features.
                point_cloud_encoded=(
                    None
                    if epoch % self.point_cloud_encoder_learning_freq == 0
                    else point_cloud_encoded_features_actor_cached[batch_idx]
                ),
                masks=masks_batch,
                hidden_states=hid_states_batch[0],
            )
            if point_cloud_encoded_features_actor_cached is None:
                # Initialize the point cloud encoded features for the first time.
                # Initialize with nan to avoid accidental use of the unintialized values.
                point_cloud_encoded_features_actor_cached = (
                    torch.ones(batch_size, *self.actor_critic.point_cloud_encoded.shape[1:]).to(self.device) * torch.nan
                )
            # Store the computed point cloud encoded features
            with torch.no_grad():
                point_cloud_encoded_features_actor_cached[batch_idx] = self.actor_critic.point_cloud_encoded.clone()
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            # -- critic
            value_batch = self.actor_critic.evaluate(
                critic_obs_batch,
                point_cloud_obs_batch,
                # If epoch % point_cloud_encoder_learning_freq == 0, then we compute the point cloud encoded features
                # using the point cloud encoder. Otherwise, we use the previously computed point cloud encoded features.
                point_cloud_encoded=(
                    None
                    if epoch % self.point_cloud_encoder_learning_freq == 0
                    else point_cloud_encoded_features_critic_cached[batch_idx]
                ),
                masks=masks_batch,
                hidden_states=hid_states_batch[1],
            )
            if point_cloud_encoded_features_critic_cached is None:
                # Initialize the point cloud encoded features for the first time
                point_cloud_encoded_features_critic_cached = (
                    torch.ones(batch_size, *self.actor_critic.point_cloud_encoded.shape[1:]).to(self.device) * torch.nan
                )
            # Store the computed point cloud encoded features
            with torch.no_grad():
                point_cloud_encoded_features_critic_cached[batch_idx] = self.actor_critic.point_cloud_encoded.clone()
            # -- entropy
            # we only keep the entropy of the first augmentation (the original one)
            mu_batch = self.actor_critic.action_mean[:original_batch_size]
            sigma_batch = self.actor_critic.action_std[:original_batch_size]
            entropy_batch = self.actor_critic.entropy[:original_batch_size]

            # KL
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            # Symmetry loss
            if self.symmetry:
                # obtain the symmetric actions
                # if we did augmentation before then we don't need to augment again
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(
                        obs=obs_batch, actions=None, env=self.symmetry["_env"], is_critic=False
                    )
                    # compute number of augmentations per sample
                    num_aug = int(obs_batch.shape[0] / original_batch_size)

                # actions predicted by the actor for symmetrically-augmented observations
                mean_actions_batch = self.actor_critic.act_inference(obs_batch.detach().clone())

                # compute the symmetrically augmented actions
                # note: we are assuming the first augmentation is the original one.
                #   We do not use the action_batch from earlier since that action was sampled from the distribution.
                #   However, the symmetry loss is computed using the mean of the distribution.
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"], is_critic=False
                )

                # compute the loss (we skip the first augmentation as it is the original one)
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
                )
                # add the loss to the total loss
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            # Random Network Distillation loss
            if self.rnd:
                # predict the embedding and the target
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch)
                # compute the loss as the mean squared error
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding.detach())

            # Gradient step
            # -- For PPO
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            # -- For RND
            if self.rnd_optimizer:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()
                self.rnd_optimizer.step()

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            # -- RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            # -- Symmetry loss
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        # -- For PPO
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        # -- For RND
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        # -- For Symmetry
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        # -- Clear the storage
        self.storage.clear()

        loss_dict = {
            "mean_value_loss": mean_value_loss,
            "mean_surrogate_loss": mean_surrogate_loss,
            "mean_entropy": mean_entropy,
            "mean_rnd_loss": mean_rnd_loss if mean_rnd_loss is not None else 0,
            "mean_symmetry_loss": mean_symmetry_loss if mean_symmetry_loss is not None else 0,
        }
        return loss_dict  # noqa: R504
