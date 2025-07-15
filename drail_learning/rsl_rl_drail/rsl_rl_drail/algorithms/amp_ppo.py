# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic
from rsl_rl_drail.storage import AMPRolloutStorage


class AMPPPO(PPO):
    """Adds discriminator training in AMP framework."""

    discriminator: ActorCritic

    def __init__(self, *args, discriminator, obs_demo, discriminator_l2_reg, discriminator_grad_penalty, **kwargs):
        super().__init__(*args, **kwargs)
        self.discriminator = discriminator
        self.discriminator.to(self.device)
        self.obs_demo = obs_demo
        self.discriminator_l2_reg = discriminator_l2_reg
        self.discriminator_grad_penalty = discriminator_grad_penalty
        self.optimizer_discriminator = optim.Adam(self.discriminator.parameters(), lr=kwargs["learning_rate"])

    def init_storage(
        self,
        training_type,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        critic_obs_shape,
        amp_obs_shape,
        action_shape,
    ):
        # create memory for RND as well :)
        if self.rnd:
            rnd_state_shape = [self.rnd.num_states]
        else:
            rnd_state_shape = None
        # create rollout storage
        self.storage = AMPRolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            amp_obs_shape,
            action_shape,
            rnd_state_shape,
            self.device,
        )

    def act(self, obs, critic_obs, amp_obs):
        self.transition.amp_observations = amp_obs
        return super().act(obs, critic_obs)

    def _disc_loss_neg(self, disc_logits):
        bce = torch.nn.BCEWithLogitsLoss()
        return bce(disc_logits, torch.zeros_like(disc_logits))

    def _disc_loss_pos(self, disc_logits):
        bce = torch.nn.BCEWithLogitsLoss()
        return bce(disc_logits, torch.ones_like(disc_logits))

    def _train_discriminator(self):
        self.discriminator.train()

        value_losses = []
        disc_agent_accs = []
        disc_demo_accs = []
        disc_agent_logits = []
        disc_demo_logits = []
        disc_grad_penalties = []
        disc_logit_reg_losses = []
        num_mini_batches = 2

        if self.discriminator.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(
                num_mini_batches=num_mini_batches, num_epochs=self.num_learning_epochs
            )
        else:
            generator = self.storage.discrim_mini_batch_generator(
                num_mini_batches=num_mini_batches, num_epochs=self.num_learning_epochs
            )

        # for processed_mini_batch in generator:
        # iterate over batches
        # TODO: Allow for recurrent discriminators?
        for (
            obs_batch,
            amp_obs_batch,
            _,
            _,
        ) in generator:
            # obs_batch = processed_mini_batch["obs_batch"]
            # amp_obs_batch = processed_mini_batch["amp_obs_batch"]

            idx = torch.randint(0, self.obs_demo.size(0), (obs_batch.size(0),))

            obs_demo = torch.autograd.Variable(self.obs_demo[idx], requires_grad=True).to(self.device)

            disc_demo_logit = self.discriminator(obs_demo)
            disc_agent_logit = self.discriminator(amp_obs_batch)

            # Classic Discriminator Loss with loss clipping for stability
            disc_loss_demo = self._disc_loss_pos(torch.clamp(disc_demo_logit, -8.0, 8.0))
            disc_loss_agent = self._disc_loss_neg(torch.clamp(disc_agent_logit, -8.0, 8.0))
            disc_loss_p1 = 0.5 * (disc_loss_agent + disc_loss_demo)

            # Discriminator weight regularization
            disc_logit_layer_weight = self.discriminator[0].weight  # weight of the first layer.
            disc_logit_reg_loss = torch.sum(torch.square(disc_logit_layer_weight))  # L2 regularization

            # Gradient penalty
            disc_demo_grad = torch.autograd.grad(
                disc_demo_logit,
                obs_demo,
                grad_outputs=torch.ones_like(disc_demo_logit),
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            disc_demo_grad = torch.sum(torch.square(disc_demo_grad), dim=-1)
            disc_grad_penalty = torch.mean(torch.clamp(disc_demo_grad, 0, 10.0))

            disc_loss = (
                disc_loss_p1
                + self.discriminator_l2_reg * disc_logit_reg_loss
                + self.discriminator_grad_penalty * disc_grad_penalty
            )

            self.optimizer_discriminator.zero_grad()
            disc_loss.backward()
            # Add gradient clipping like in PPO
            max_grad_norm = 1.0
            nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_grad_norm)
            self.optimizer_discriminator.step()

            # Calculate accuracies
            agent_acc = (disc_agent_logit < 0).float().mean()
            demo_acc = (disc_demo_logit > 0).float().mean()

            # Store metrics
            value_losses.append(disc_loss.item())
            disc_agent_accs.append(agent_acc.item())
            disc_demo_accs.append(demo_acc.item())
            disc_agent_logits.append(disc_agent_logit.mean().item())
            disc_demo_logits.append(disc_demo_logit.mean().item())
            disc_grad_penalties.append(disc_grad_penalty.item())
            disc_logit_reg_losses.append(disc_logit_reg_loss.item())

        return {
            "disc_mean_loss": np.mean(value_losses),
            "disc_agent_acc": np.mean(disc_agent_accs),
            "disc_demo_acc": np.mean(disc_demo_accs),
            "disc_agent_logit": np.mean(disc_agent_logits),
            "disc_demo_logit": np.mean(disc_demo_logits),
            "disc_grad_penalty": np.mean(disc_grad_penalties),
            "disc_logit_reg_loss": np.mean(disc_logit_reg_losses),
        }

    def update(self):
        # Get discriminator losses if training discriminator
        disc_losses = self._train_discriminator() if self.train_discriminator_flag else {}
        # Get base PPO losses from parent class
        # We need to explicitly pass self to super() in this case
        ppo_losses = super().update()
        # Merge the two dictionaries
        return {**ppo_losses, **disc_losses}
