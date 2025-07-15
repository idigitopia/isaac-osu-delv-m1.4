# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import inspect

import rsl_rl.algorithms as rsl_rl_algorithms


class PPO(rsl_rl_algorithms.PPO):
    """Allow extra kwargs to be passed and reports the unexpected kwargs. This is to make it
    consistent with the ActorCritic class."""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        # Get the parent class's __init__ signature
        parent_sig = inspect.signature(rsl_rl_algorithms.PPO.__init__)
        parent_params = set(parent_sig.parameters.keys()) - {"self"}

        # Check for extra kwargs that the parent class doesn't accept
        extra_kwargs = set(kwargs.keys()) - parent_params
        if extra_kwargs:
            print(f"Warning: PPO.__init__ received unexpected keyword arguments: {extra_kwargs}")

        # Filter kwargs to only include valid parent parameters
        valid_kwargs = {k: v for k, v in kwargs.items() if k in parent_params}

        super().__init__(*args, **valid_kwargs)
