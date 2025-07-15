# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def modify_cmd_range(env: ManagerBasedRLEnv, env_ids: Sequence[int],
                    term_name: str, range_name: str, range_set: tuple[float, float], num_steps: int):
    """Curriculum that modifies a command sampling range at a given number of steps.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the reward term.
        range_name: The name of the range to be modified.
        range_set: The range to be modified.
        num_steps: The number of steps after which the change should be applied.
    """
    if env.common_step_counter > num_steps:
        # obtain term settings
        term = env.command_manager.get_term(term_name)

        # set the range
        if hasattr(term.cfg.ranges, range_name):
            setattr(term.cfg.ranges, range_name, range_set)
        else:
            raise RuntimeError(f"Command term {term_name} does not have range {range_name}.")


def modify_cmd_sampling_prob(env: ManagerBasedRLEnv, env_ids: Sequence[int],
                    term_name: str, prob_name: str, prob_val: float, num_steps: int):
    """Curriculum that modifies a command sampling probability at a given number of steps.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the reward term.
        prob_name: The name of the probability term in the cfg.
        prob_val: The value to be modified.
        num_steps: The number of steps after which the change should be applied.
    """
    if env.common_step_counter > num_steps:
        # obtain term settings
        term = env.command_manager.get_term(term_name)

        # get the term ranges
        if hasattr(term.cfg, prob_name):
            setattr(term.cfg, prob_name, prob_val)
        else:
            raise RuntimeError(f"Command term {term_name} does not have {prob_name}.")


def modify_cmd_category_prob(env: ManagerBasedRLEnv, env_ids: Sequence[int],
                    term_name: str, prob_vals: list[float], num_steps: int):
    """Curriculum that modifies a command sampling probability at a given number of steps.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the reward term.
        prob_vals: A list of the values to be modified. Make sure the list has the same length as the
            number of categories in the command term.
        num_steps: The number of steps after which the change should be applied.
    """
    if env.common_step_counter > num_steps:
        # obtain term settings
        term = env.command_manager.get_term(term_name)

        if hasattr(term, "ranges_prob"):

            # check if the number of categories is the same as the provided probabilities
            assert len(prob_vals) == term.ranges_prob.shape[0], \
                f"Command term {term_name} does not have the same number of categories as the provided probabilities."

            # set the probabilities
            term.ranges_prob = torch.tensor(prob_vals, device=env.device, dtype=torch.float32)
            term.ranges_prob = term.ranges_prob / torch.sum(term.ranges_prob)

        else:
            raise RuntimeError(f"Command term {term_name} does not have ranges_prob.")
