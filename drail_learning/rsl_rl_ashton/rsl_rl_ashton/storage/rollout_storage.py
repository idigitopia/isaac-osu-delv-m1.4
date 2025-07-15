# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from rsl_rl_ashton.utils import split_and_pad_trajectories
from collections import defaultdict


class RolloutStorage:
    class Transition:
        def __init__(self):
            self.observations = None
            self.critic_observations = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.hidden_states = None
            self.rnd_state = None

        def clear(self):
            self.__init__()

    def __init__(
        self,
        num_envs,
        num_transitions_per_env,
        obs_shape,
        privileged_obs_shape,
        actions_shape,
        rnd_state_shape=None,
        device="cpu",
    ):
        # store inputs
        self.device = device
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.obs_shape = obs_shape
        self.privileged_obs_shape = privileged_obs_shape
        self.rnd_state_shape = rnd_state_shape
        self.actions_shape = actions_shape

        # Initialize tensor dictionary
        self.setup_transition_buffer()

        # counter for the number of transitions stored
        self.step = 0

    def setup_transition_buffer(self):
        self.experience_buffer = {
            # Core
            'observations': torch.zeros(self.num_transitions_per_env, self.num_envs, *self.obs_shape, device=self.device),
            'actions': torch.zeros(self.num_transitions_per_env, self.num_envs, *self.actions_shape, device=self.device),
            'rewards': torch.zeros(self.num_transitions_per_env, self.num_envs, 1, device=self.device),
            'dones': torch.zeros(self.num_transitions_per_env, self.num_envs, 1, device=self.device).byte(),

            # For PPO
            'values': torch.zeros(self.num_transitions_per_env, self.num_envs, 1, device=self.device),
            'returns': torch.zeros(self.num_transitions_per_env, self.num_envs, 1, device=self.device),
            'advantages': torch.zeros(self.num_transitions_per_env, self.num_envs, 1, device=self.device),
            'actions_log_prob': torch.zeros(self.num_transitions_per_env, self.num_envs, 1, device=self.device),
            'mu': torch.zeros(self.num_transitions_per_env, self.num_envs, *self.actions_shape, device=self.device),
            'sigma': torch.zeros(self.num_transitions_per_env, self.num_envs, *self.actions_shape, device=self.device),
        }

        if self.privileged_obs_shape is not None:
            self.experience_buffer['privileged_observations'] = torch.zeros(self.num_transitions_per_env, self.num_envs, *self.privileged_obs_shape, device=self.device)

        if self.rnd_state_shape is not None:
            self.experience_buffer['rnd_state'] = torch.zeros(self.num_transitions_per_env, self.num_envs, *self.rnd_state_shape, device=self.device)

        # For RNN networks
        self.saved_hidden_states_a = None
        self.saved_hidden_states_c = None
        self.observation_keys = ['observations', 'privileged_observations', 'rnd_state']

    def add_transitions(self, transition: Transition):
        # check if the transition is valid
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow! You should call clear() before adding new transitions.")

        # Core
        self.experience_buffer['observations'][self.step].copy_(transition.observations)
        self.experience_buffer['actions'][self.step].copy_(transition.actions)
        self.experience_buffer['rewards'][self.step].copy_(transition.rewards.view(-1, 1))
        self.experience_buffer['dones'][self.step].copy_(transition.dones.view(-1, 1))
        if 'privileged_observations' in self.experience_buffer:
            self.experience_buffer['privileged_observations'][self.step].copy_(transition.critic_observations)

        # For PPO
        self.experience_buffer['values'][self.step].copy_(transition.values)
        self.experience_buffer['actions_log_prob'][self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.experience_buffer['mu'][self.step].copy_(transition.action_mean)
        self.experience_buffer['sigma'][self.step].copy_(transition.action_sigma)

        # For RND
        if 'rnd_state' in self.experience_buffer:
            self.experience_buffer['rnd_state'][self.step].copy_(transition.rnd_state)

        # For RNN networks
        self._save_hidden_states(transition.hidden_states)

        # increment the counter
        self.step += 1

    def _save_hidden_states(self, hidden_states):
        if hidden_states is None or hidden_states == (None, None):
            return
        # make a tuple out of GRU hidden state sto match the LSTM format
        hid_a = hidden_states[0] if isinstance(hidden_states[0], tuple) else (hidden_states[0],)
        hid_c = hidden_states[1] if isinstance(hidden_states[1], tuple) else (hidden_states[1],)

        # initialize if needed
        if self.saved_hidden_states_a is None:
            self.saved_hidden_states_a = [
                torch.zeros(self.experience_buffer['observations'].shape[0], *hid_a[i].shape, device=self.device) for i in range(len(hid_a))
            ]
            self.saved_hidden_states_c = [
                torch.zeros(self.experience_buffer['observations'].shape[0], *hid_c[i].shape, device=self.device) for i in range(len(hid_c))
            ]
        # copy the states
        for i in range(len(hid_a)):
            self.saved_hidden_states_a[i][self.step].copy_(hid_a[i])
            self.saved_hidden_states_c[i][self.step].copy_(hid_c[i])

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, gamma, lam):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            # if we are at the last step, bootstrap the return value
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.experience_buffer['values'][step + 1]

            # 1 if we are not in a terminal state, 0 otherwise
            next_is_not_terminal = 1.0 - self.experience_buffer['dones'][step].float()
            # TD error: r_t + gamma * V(s_{t+1}) - V(s_t)
            delta = (
                self.experience_buffer['rewards'][step] +
                next_is_not_terminal * gamma * next_values -
                self.experience_buffer['values'][step]
            )
            # Advantage: A(s_t, a_t) = delta_t + gamma * lambda * A(s_{t+1}, a_{t+1})
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.experience_buffer['returns'][step] = advantage + self.experience_buffer['values'][step]

        # Compute and normalize advantages
        self.experience_buffer['advantages'] = self.experience_buffer['returns'] - self.experience_buffer['values']
        self.experience_buffer['advantages'] = (
            self.experience_buffer['advantages'] - self.experience_buffer['advantages'].mean()
        ) / (self.experience_buffer['advantages'].std() + 1e-8)

    def get_statistics(self):
        done = self.experience_buffer['dones']
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat(
            (flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0])
        )
        trajectory_lengths = done_indices[1:] - done_indices[:-1]
        return trajectory_lengths.float().mean(), self.experience_buffer['rewards'].mean()


    # def mini_batch_generator(self, num_mini_batches, num_epochs=8):
    #     batch_size = self.num_envs * self.num_transitions_per_env
    #     mini_batch_size = batch_size // num_mini_batches
    #     indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

    #     # Core
    #     observations = self.observations.flatten(0, 1)
    #     if self.privileged_observations is not None:
    #         critic_observations = self.privileged_observations.flatten(0, 1)
    #     else:
    #         critic_observations = observations

    #     actions = self.actions.flatten(0, 1)
    #     values = self.values.flatten(0, 1)
    #     returns = self.returns.flatten(0, 1)

    #     # For PPO
    #     old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
    #     advantages = self.advantages.flatten(0, 1)
    #     old_mu = self.mu.flatten(0, 1)
    #     old_sigma = self.sigma.flatten(0, 1)

    #     # For RND
    #     if self.rnd_state_shape is not None:
    #         rnd_state = self.rnd_state.flatten(0, 1)

    #     for epoch in range(num_epochs):
    #         for i in range(num_mini_batches):
    #             # Select the indices for the mini-batch
    #             start = i * mini_batch_size
    #             end = (i + 1) * mini_batch_size
    #             batch_idx = indices[start:end]

    #             # Create the mini-batch
    #             # -- Core
    #             obs_batch = observations[batch_idx]
    #             critic_observations_batch = critic_observations[batch_idx]
    #             actions_batch = actions[batch_idx]

    #             # -- For PPO
    #             target_values_batch = values[batch_idx]
    #             returns_batch = returns[batch_idx]
    #             old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
    #             advantages_batch = advantages[batch_idx]
    #             old_mu_batch = old_mu[batch_idx]
    #             old_sigma_batch = old_sigma[batch_idx]

    #             # -- For RND
    #             if self.rnd_state_shape is not None:
    #                 rnd_state_batch = rnd_state[batch_idx]
    #             else:
    #                 rnd_state_batch = None

    #             # Yield the mini-batch
    #             yield obs_batch, critic_observations_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (
    #                 None,
    #                 None,
    #             ), None, rnd_state_batch



    def raw_mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        flattened_dict = {}
        for key in self.experience_buffer.keys():
            flattened_dict[key] = self.experience_buffer[key].flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = start + mini_batch_size
                batch_idx = indices[start:end]

                yield {
                    **{key: value[batch_idx] for key, value in flattened_dict.items()},
                    'epoch': epoch,
                    'batch_size': len(indices),
                    'batch_idx': batch_idx
                }

    def process_raw_mini_batch(self, raw_mini_batch):
        processed_mini_batch = {}
        processed_mini_batch["obs_batch"] = raw_mini_batch['observations']
        processed_mini_batch["critic_obs_batch"] = raw_mini_batch.get('privileged_observations', raw_mini_batch['observations'])
        processed_mini_batch["actions_batch"] = raw_mini_batch['actions']
        processed_mini_batch["target_values_batch"] = raw_mini_batch['values']
        processed_mini_batch["returns_batch"] = raw_mini_batch['returns']
        processed_mini_batch["advantages_batch"] = raw_mini_batch['advantages']
        processed_mini_batch["old_actions_log_prob_batch"] = raw_mini_batch['actions_log_prob']
        processed_mini_batch["old_mu_batch"] = raw_mini_batch['mu']
        processed_mini_batch["old_sigma_batch"] = raw_mini_batch['sigma']
        processed_mini_batch["rnd_state_batch"] = raw_mini_batch.get('rnd_state', None)
        processed_mini_batch["hid_a"] = raw_mini_batch.get("hid_a", None)
        processed_mini_batch["hid_c"] = raw_mini_batch.get("hid_c", None)
        processed_mini_batch["obs_mask_batch"] = raw_mini_batch.get("masks_observations", None)
        processed_mini_batch["epoch"] = raw_mini_batch.get("epoch", None)
        processed_mini_batch["batch_size"] = raw_mini_batch.get("batch_size", None)
        processed_mini_batch["batch_idx"] = raw_mini_batch.get("batch_idx", None)
        return processed_mini_batch

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        raw_mini_batch_generator = self.raw_mini_batch_generator(num_mini_batches, num_epochs)
        for raw_mini_batch in raw_mini_batch_generator:
            yield self.process_raw_mini_batch(raw_mini_batch)


    from collections import defaultdict

    def raw_recurrent_mini_batch_generator(self, num_mini_batches, num_epochs=8):
        padded_obs_trajectories_dict = defaultdict(lambda: {})
        padded_obs_trajectoires_masks_dict = defaultdict(lambda: {})
        for key in self.observation_keys:
            if key in self.experience_buffer:
                trajectories, trajectory_masks = split_and_pad_trajectories(self.experience_buffer[key], self.experience_buffer['dones'])
                padded_obs_trajectories_dict[key] = trajectories
                padded_obs_trajectoires_masks_dict[key] = trajectory_masks

        raw_mini_batch = {}
        mini_batch_size = self.num_envs // num_mini_batches
        for ep in range(num_epochs):
            first_traj = 0
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size

                dones = self.experience_buffer['dones'].squeeze(-1)
                last_was_done = torch.zeros_like(dones, dtype=torch.bool)
                last_was_done[1:] = dones[:-1]
                last_was_done[0] = True
                trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
                last_traj = first_traj + trajectories_batch_size

                # masks_batch = padded_obs_trajectoires_masks_dict['observations'][:, first_traj:last_traj]
                # obs_batch = padded_obs_trajectories_dict['observations'][:, first_traj:last_traj]
                # critic_obs_batch = padded_obs_trajectories_dict['observations'][:, first_traj:last_traj]

                for key in self.experience_buffer:
                    if key in self.observation_keys:
                        raw_mini_batch["padded_" + key] = padded_obs_trajectories_dict[key][:, first_traj:last_traj]
                        raw_mini_batch["masks_" + key] = padded_obs_trajectoires_masks_dict[key][:, first_traj:last_traj]
                    else:
                        raw_mini_batch[key] = self.experience_buffer[key][:, start:stop]


                # NOTES:
                #
                # Hidden State Initialization Scenarios:
                #
                # 1. New Trajectory After Termination/Truncation:
                #    - When a trajectory ends due to termination or truncation (indicated by `done = True`),
                #      the hidden state for the subsequent trajectory is reset to the model's default initial state.
                #
                # 2. New Trajectory After Reaching Maximum Transitions in Rollout Loop (num_transitions_per_env):
                #    - If a rollout loop reaches the maximum number of transitions in any iteration, environment
                #      that has not terminated or truncated will have its hidden state carried over from the last state
                #      of that trajectory in the previous iteration.
                #
                # This operation is performed by using a mask "last_was_done" which is obtained by shifting the dones
                # tensor by one position to the right and setting the first element to True. Shifing is done to indicate
                # that after done, the next trajectory has started. And, we set first element of last_was_done to True
                # to indicate the continuation of the trajectory from last iteration.

                # reshape to [num_envs, time, num layers, hidden dim] (original shape: [time, num_layers, num_envs, hidden_dim])
                # then take only time steps after dones (flattens num envs and time dimensions),
                # take a batch of trajectories and finally reshape back to [num_layers, batch, hidden_dim]
                last_was_done = last_was_done.permute(1, 0)
                hid_a_batch = [
                    saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
                    .transpose(1, 0)
                    .contiguous()
                    for saved_hidden_states in self.saved_hidden_states_a
                ]
                hid_c_batch = [
                    saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
                    .transpose(1, 0)
                    .contiguous()
                    for saved_hidden_states in self.saved_hidden_states_c
                ]
                # remove the tuple for GRU
                hid_a_batch = hid_a_batch[0] if len(hid_a_batch) == 1 else hid_a_batch
                hid_c_batch = hid_c_batch[0] if len(hid_c_batch) == 1 else hid_c_batch

                raw_mini_batch["hid_a"] = hid_a_batch
                raw_mini_batch["hid_c"] = hid_c_batch

                yield raw_mini_batch

                first_traj = last_traj

    def process_raw_recurrent_mini_batch(self, raw_mini_batch):
        processed_mini_batch = {}
        processed_mini_batch["obs_batch"] = raw_mini_batch["padded_observations"]
        processed_mini_batch["critic_obs_batch"] = raw_mini_batch.get('padded_privileged_observations', raw_mini_batch["padded_observations"])
        processed_mini_batch["actions_batch"] = raw_mini_batch["actions"]
        processed_mini_batch["target_values_batch"] = raw_mini_batch["values"]
        processed_mini_batch["returns_batch"] = raw_mini_batch["returns"]
        processed_mini_batch["advantages_batch"] = raw_mini_batch["advantages"]
        processed_mini_batch["old_actions_log_prob_batch"] = raw_mini_batch["actions_log_prob"]
        processed_mini_batch["old_mu_batch"] = raw_mini_batch["mu"]
        processed_mini_batch["old_sigma_batch"] = raw_mini_batch["sigma"]
        processed_mini_batch["rnd_state_batch"] = raw_mini_batch.get('padded_rnd_state', None)
        processed_mini_batch["hid_a"] = raw_mini_batch.get("hid_a", None)
        processed_mini_batch["hid_c"] = raw_mini_batch.get("hid_c", None)
        processed_mini_batch["obs_mask_batch"] = raw_mini_batch.get("masks_observations", None)

        return processed_mini_batch

    def recurrent_mini_batch_generator(self, num_mini_batches, num_epochs=8):
        raw_recurrent_mini_batch_generator = self.raw_recurrent_mini_batch_generator(num_mini_batches, num_epochs)
        for raw_mini_batch in raw_recurrent_mini_batch_generator:
            yield self.process_raw_recurrent_mini_batch(raw_mini_batch)

    # for RNNs only
    # def recurrent_mini_batch_generator(self, num_mini_batches, num_epochs=8):
    #     padded_obs_trajectories, trajectory_masks = split_and_pad_trajectories(self.experience_buffer['observations'], self.experience_buffer['dones'])
    #     if 'privileged_observations' in self.experience_buffer:
    #         padded_critic_obs_trajectories, _ = split_and_pad_trajectories(self.experience_buffer['privileged_observations'], self.experience_buffer['dones'])
    #     else:
    #         padded_critic_obs_trajectories = padded_obs_trajectories

    #     if 'rnd_state' in self.experience_buffer:
    #         padded_rnd_state_trajectories, _ = split_and_pad_trajectories(self.experience_buffer['rnd_state'], self.experience_buffer['dones'])
    #     else:
    #         padded_rnd_state_trajectories = None

    #     mini_batch_size = self.num_envs // num_mini_batches
    #     for ep in range(num_epochs):
    #         first_traj = 0
    #         for i in range(num_mini_batches):
    #             start = i * mini_batch_size
    #             stop = (i + 1) * mini_batch_size

    #             dones = self.experience_buffer['dones'].squeeze(-1)
    #             last_was_done = torch.zeros_like(dones, dtype=torch.bool)
    #             last_was_done[1:] = dones[:-1]
    #             last_was_done[0] = True
    #             trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
    #             last_traj = first_traj + trajectories_batch_size

    #             masks_batch = trajectory_masks[:, first_traj:last_traj]
    #             obs_batch = padded_obs_trajectories[:, first_traj:last_traj]
    #             critic_obs_batch = padded_critic_obs_trajectories[:, first_traj:last_traj]

    #             if padded_rnd_state_trajectories is not None:
    #                 rnd_state_batch = padded_rnd_state_trajectories[:, first_traj:last_traj]
    #             else:
    #                 rnd_state_batch = None

    #             actions_batch = self.experience_buffer['actions'][:, start:stop]
    #             old_mu_batch = self.experience_buffer['mu'][:, start:stop]
    #             old_sigma_batch = self.experience_buffer['sigma'][:, start:stop]
    #             returns_batch = self.experience_buffer['returns'][:, start:stop]
    #             advantages_batch = self.experience_buffer['advantages'][:, start:stop]
    #             values_batch = self.experience_buffer['values'][:, start:stop]
    #             old_actions_log_prob_batch = self.experience_buffer['actions_log_prob'][:, start:stop]

    #             # reshape to [num_envs, time, num layers, hidden dim] (original shape: [time, num_layers, num_envs, hidden_dim])
    #             # then take only time steps after dones (flattens num envs and time dimensions),
    #             # take a batch of trajectories and finally reshape back to [num_layers, batch, hidden_dim]
    #             last_was_done = last_was_done.permute(1, 0)
    #             hid_a_batch = [
    #                 saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
    #                 .transpose(1, 0)
    #                 .contiguous()
    #                 for saved_hidden_states in self.saved_hidden_states_a
    #             ]
    #             hid_c_batch = [
    #                 saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
    #                 .transpose(1, 0)
    #                 .contiguous()
    #                 for saved_hidden_states in self.saved_hidden_states_c
    #             ]
    #             # remove the tuple for GRU
    #             hid_a_batch = hid_a_batch[0] if len(hid_a_batch) == 1 else hid_a_batch
    #             hid_c_batch = hid_c_batch[0] if len(hid_c_batch) == 1 else hid_c_batch

    #             yield obs_batch, critic_obs_batch, actions_batch, values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (
    #                 hid_a_batch,
    #                 hid_c_batch,
    #             ), masks_batch, rnd_state_batch

    #             first_traj = last_traj
