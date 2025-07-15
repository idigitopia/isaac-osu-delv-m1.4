# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for the 2D-pose for locomotion tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .gui_keyboard_command_cfg import (
        GUIKeyboardVelocityCommandCfg,
        GUIKeyboardWholeBodyCfg,
        GUIKeyboardPoseCfg,
    )


from drail_extensions.research_ashton.mdp import UniformVelocityCommand, UniformPoseCommand

from .whole_body_command import CategoricalCommand
from .gui_keyboard_se3_command import (
    GUIKeyboardSE3Command,
    GUIKeyboardSE3WholeBodyCommand,
    GUIKeyboardSE3PoseCommand,
)



class GUIKeyboardVelocityCommand(GUIKeyboardSE3Command, UniformVelocityCommand):
    r"""Command generator that generates a velocity command based on keyboard input."""

    cfg: GUIKeyboardVelocityCommandCfg
    """The configuration of the command generator."""

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "GUIKeyboardVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        return msg

    def _set_debug_vis_impl(self, debug_vis: bool):
        UniformVelocityCommand._set_debug_vis_impl(self, debug_vis)

    def _debug_vis_callback(self, event):
        return UniformVelocityCommand._debug_vis_callback(self, event)

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return UniformVelocityCommand._resolve_xy_velocity_to_arrow(self, xy_velocity)


class GUIKeyboardWholeBodyCommand(GUIKeyboardSE3WholeBodyCommand, CategoricalCommand):
    r"""Command generator that generates a velocity/wholebody command based on keyboard input."""

    cfg: GUIKeyboardWholeBodyCfg
    """The configuration of the command generator."""

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "GUIKeyboardVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        return msg

    def _set_debug_vis_impl(self, debug_vis: bool):
        CategoricalCommand._set_debug_vis_impl(self, debug_vis)

    def _debug_vis_callback(self, event):
        return CategoricalCommand._debug_vis_callback(self, event)

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return CategoricalCommand._resolve_xy_velocity_to_arrow(self, xy_velocity)


class GUIKeyboardPoseCommand(GUIKeyboardSE3PoseCommand, UniformPoseCommand):
    r"""Command generator that generates a SE3 Pose command based on keyboard input."""

    cfg: GUIKeyboardPoseCfg
    """The configuration of the command generator."""

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "GUIKeyboardVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        return msg

    def _set_debug_vis_impl(self, debug_vis: bool):
        UniformPoseCommand._set_debug_vis_impl(self, debug_vis)
        # self.goal_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        return UniformPoseCommand._debug_vis_callback(self, event)