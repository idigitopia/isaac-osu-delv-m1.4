from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
import torch
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from . import actions_cfg


class JointPositionAction_v2(mdp.JointPositionAction):
    """Add a functionality of saving previous raw actions per term."""

    cfg: actions_cfg.JointPositionAction_v2_Cfg

    def process_actions(self, actions: torch.Tensor):
        # apply the action
        self._previous_raw_actions = self._raw_actions.clone()
        super().process_actions(actions)

    @property
    def previous_raw_actions(self) -> torch.Tensor:
        return self._previous_raw_actions


@configclass
class JointPositionAction_v2_Cfg(mdp.JointPositionActionCfg):
    """Add a functionality of saving previous raw actions per term."""

    class_type: type[ActionTerm] = JointPositionAction_v2
