# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.storage import AMPRolloutStorage
from rsl_rl.algorithms import PPO 
from rsl_rl.modules import ActorCritic
import numpy as np
import math

class FinetunePPO(PPO):
    """Adds discriminator traiing in AMP framework."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def freeze_critic(self):
        for param in self.actor_critic.critic.parameters():
            param.requires_grad = False

    def unfreeze_critic(self):
        for param in self.actor_critic.critic.parameters():
            param.requires_grad = True

    def freeze_actor(self):
        for param in self.actor_critic.actor.parameters():
            param.requires_grad = False

    def unfreeze_actor(self):
        for param in self.actor_critic.actor.parameters():
            param.requires_grad = True  

    def reset_actor(self):
        # reinitialize actor parameters using default PyTorch initialization
        for layer in self.actor_critic.actor:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
                
    def reset_critic(self):
        # reinitialize critic parameters using default PyTorch initialization
        for layer in self.actor_critic.critic:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
                    
    def refresh_optimizer(self):
        assert hasattr(self, 'optimizer')
        learnable_params = filter(lambda p: p.requires_grad, self.actor_critic.parameters())
        self.optimizer = optim.Adam(learnable_params, lr= self.optimizer.defaults['lr'])
