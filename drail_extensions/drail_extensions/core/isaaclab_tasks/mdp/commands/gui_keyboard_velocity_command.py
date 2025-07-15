# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for the 2D-pose for locomotion tasks."""

from __future__ import annotations

import torch
from isaaclab.envs.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils import configclass

from .gui_keyboard_se3_command import GUIKeyboardSE3Command, GUIKeyboardSE3CommandCfg


class GUIKeyboardVelocityCommand(GUIKeyboardSE3Command, UniformVelocityCommand):
    r"""Command generator that generates a velocity command based on keyboard input."""

    cfg: GUIKeyboardVelocityCommandCfg
    """The configuration of the command generator."""

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "GUIKeyboardVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, 3)."""
        return super().command[:, :3]

    def _set_debug_vis_impl(self, debug_vis: bool):
        UniformVelocityCommand._set_debug_vis_impl(self, debug_vis)

    def _debug_vis_callback(self, event):
        return UniformVelocityCommand._debug_vis_callback(self, event)

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return UniformVelocityCommand._resolve_xy_velocity_to_arrow(self, xy_velocity)


@configclass
class GUIKeyboardVelocityCommandCfg(GUIKeyboardSE3CommandCfg, UniformVelocityCommandCfg):
    """Configuration for the velocity command based on keyboard input."""

    class_type: type = GUIKeyboardVelocityCommand

    # Dummy and unused range as UniformVelocityCommand __init__() requires a range
    ranges: UniformVelocityCommandCfg.Ranges = UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-0.0, 0.0), lin_vel_y=(-0.0, 0.0), ang_vel_z=(-0.0, 0.0), heading=(-0.0, 0.0)
    )
