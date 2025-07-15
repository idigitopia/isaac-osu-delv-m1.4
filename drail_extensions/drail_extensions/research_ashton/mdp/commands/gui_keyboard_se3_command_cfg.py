# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass

from .gui_keyboard_se3_command import *


@configclass
class GUIKeyboardSE3CommandCfg(CommandTermCfg):
    """Configuration for the SE(3) command based on keyboard input."""

    class_type: type = GUIKeyboardSE3Command

    pos_sensitivity: float = 0.05
    """The sensitivity for the position command. Defaults to 0.05."""

    rot_sensitivity: float = 0.01
    """The sensitivity for the rotation command. Defaults to 0.05."""

    resampling_time_range: tuple[float, float] = (0.0, 0.0)
    """The time range for the resampling of the command. Defaults to (0.0, 0.0). This quantity must be set to
    (0.0, 0.0) to listen to the keyboard input at every time step. To control sensitivity,
    use the :attr:`pos_sensitivity` and :attr:`rot_sensitivity` parameters."""


@configclass
class GUIKeyboardSE3CommandStandBitCfg(CommandTermCfg):
    """Configuration for the SE(3) command based on keyboard input."""

    class_type: type = GUIKeyboardSE3CommandStandBit

    pos_sensitivity: float = 0.05
    """The sensitivity for the position command. Defaults to 0.05."""

    rot_sensitivity: float = 0.05
    """The sensitivity for the rotation command. Defaults to 0.05."""

    resampling_time_range: tuple[float, float] = (0.0, 0.0)
    """The time range for the resampling of the command. Defaults to (0.0, 0.0). This quantity must be set to
    (0.0, 0.0) to listen to the keyboard input at every time step. To control sensitivity,
    use the :attr:`pos_sensitivity` and :attr:`rot_sensitivity` parameters."""


@configclass
class GUIKeyboardSE3HeightCommandCfg(CommandTermCfg):
    """Configuration for the SE(3) command based on keyboard input."""

    class_type: type = GUIKeyboardHeightCommand

    pos_sensitivity: float = 0.001
    """The sensitivity for the position command. Defaults to 0.05."""

    rot_sensitivity: float = 0.001
    """The sensitivity for the rotation command. Defaults to 0.05."""

    resampling_time_range: tuple[float, float] = (0.0, 0.0)
    """The time range for the resampling of the command. Defaults to (0.0, 0.0). This quantity must be set to
    (0.0, 0.0) to listen to the keyboard input at every time step. To control sensitivity,
    use the :attr:`pos_sensitivity` and :attr:`rot_sensitivity` parameters."""