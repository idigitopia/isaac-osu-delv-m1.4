# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
import torch

if TYPE_CHECKING:
    from . import actions_cfg


class JointPositionAction(mdp.JointPositionAction):
    """Add a functionality of saving previous raw actions per term."""

    cfg: actions_cfg.JointPositionActionCfg

    def process_actions(self, actions: torch.Tensor):
        # apply the action
        self._previous_raw_actions = self._raw_actions.clone()
        super().process_actions(actions)

    @property
    def previous_raw_actions(self) -> torch.Tensor:
        return self._previous_raw_actions
