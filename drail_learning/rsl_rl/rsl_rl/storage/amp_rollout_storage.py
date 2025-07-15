#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
from rsl_rl.storage import RolloutStorage as BaseRolloutStorage

class AMPRolloutStorage(BaseRolloutStorage):
    class Transition(BaseRolloutStorage.Transition):
        def __init__(self):
            super().__init__()
            self.amp_observations = None

        def clear(self):
            self.__init__()

    def __init__(self, num_envs, 
                 num_transitions_per_env,
                obs_shape, 
                privileged_obs_shape, 
                amp_obs_shape, 
                actions_shape, 
                rnd_state_shape=None,
                device="cpu"):
        # add amp observations to the observation shape 
        self.amp_obs_shape = amp_obs_shape
        
        super().__init__(num_envs, num_transitions_per_env, obs_shape, privileged_obs_shape, actions_shape, rnd_state_shape, device)


    def setup_transition_buffer(self):
        super().setup_transition_buffer()
        self.experience_buffer['amp_observations'] = torch.zeros(self.num_transitions_per_env, self.num_envs, *self.amp_obs_shape, device=self.device)
        self.observation_keys.append('amp_observations')
    
    def add_transitions(self, transition: Transition):
        self.experience_buffer['amp_observations'][self.step].copy_(transition.amp_observations)
        super().add_transitions(transition)

    def process_raw_mini_batch(self, raw_mini_batch):
        processed_mini_batch = super().process_raw_mini_batch(raw_mini_batch)
        processed_mini_batch['amp_obs_batch'] = raw_mini_batch['amp_observations']
        return processed_mini_batch

    def process_raw_recurrent_mini_batch(self, raw_mini_batch):
        processed_mini_batch = super().process_raw_recurrent_mini_batch(raw_mini_batch)
        processed_mini_batch['amp_obs_batch'] = raw_mini_batch['amp_observations']
        return processed_mini_batch