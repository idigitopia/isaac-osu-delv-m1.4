# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import math
import statistics
import time
import torch
from collections import deque

from rsl_rl.algorithms import FinetunePPO
from rsl_rl.env import VecEnv
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent, EmpiricalNormalization
from rsl_rl.utils import store_code_state
import pickle

import torch.optim as optim

import rsl_rl.runners as base_runners

import isaaclab


class FinetuneOnPolicyRunner(base_runners.OnPolicyRunner):
    """On-policy runner for training and evaluation."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        assert train_cfg['algorithm']['class_name'] == 'FinetunePPO'
        super().__init__(env, train_cfg, log_dir, device)
        assert type(self.alg) == FinetunePPO

    def setup_algorithm_for_finetuning(self):
        if self.cfg['reset_actor']:
            self.alg.reset_actor()
            self.alg.refresh_optimizer()
            print("Reset actor")

        if self.cfg['reset_critic']:
            self.alg.reset_critic()   
            self.alg.refresh_optimizer()
            print("Reset critic")

        if self.cfg['freeze_critic']:
            self.alg.freeze_critic()
            self.alg.refresh_optimizer()
            print("Froze critic, and refreshed optimizer")
        
        if self.cfg['freeze_actor']:
            self.alg.freeze_actor()
            self.alg.refresh_optimizer()
            print("Froze actor, and refreshed optimizer")


    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        
        self.setup_algorithm_for_finetuning()
        
        super().learn(num_learning_iterations, init_at_random_ep_len)
    

