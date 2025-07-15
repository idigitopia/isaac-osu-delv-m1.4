# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.utils import configclass

from . import joint_actions
from . import pre_trained_policy_action

from isaaclab.managers import (
    ActionTerm,
    ActionTermCfg,
    ObservationGroupCfg,
)


@configclass
class JointPositionActionCfg2(mdp.JointPositionActionCfg):
    """Add a functionality of saving previous raw actions per term."""

    class_type: type[ActionTerm] = joint_actions.JointPositionAction


@configclass
class PreTrainedPolicyActionCfg(mdp.ActionTermCfg):
    """Configuration for pre-trained policy action term.

    See :class:`PreTrainedPolicyAction` for more details.
    """

    class_type: type[ActionTerm] = pre_trained_policy_action.PreTrainedPolicyAction
    """ Class of the action term."""
    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""
    policy_path: str = MISSING
    """Path to the low level policy (.pt files)."""
    action_dim: int = MISSING
    """Dimension of the high level action."""
    low_level_decimation: int = 4
    """Decimation factor for the low level action term."""
    low_level_actions: mdp.ActionTermCfg = MISSING
    """Low level action configuration."""
    low_level_observations: ObservationGroupCfg = MISSING
    """Low level observation configuration."""
    low_level_observations_command_name: str = MISSING
    """Name of the command to use for the low level observations."""
    action_threshold: float = 0.01
    """Action value below which the action is considered to be zero."""
    debug_vis: bool = False
    """Whether to visualize debug information. Defaults to False."""
