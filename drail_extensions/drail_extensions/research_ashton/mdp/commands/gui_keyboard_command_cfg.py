# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab.utils import configclass

from .whole_body_command_cfg import *

from .gui_keyboard_se3_command_cfg import *
from .gui_keyboard_command import *

from drail_extensions.research_ashton.mdp import UniformPoseCommandCfg


@configclass
class GUIKeyboardVelocityCommandCfg(GUIKeyboardSE3CommandCfg, UniformVelocityCommandCfg):
    """Configuration for the velocity command based on keyboard input."""

    class_type: type = GUIKeyboardVelocityCommand

    # Dummy and unused range as UniformVelocityCommand __init__() requires a range
    ranges: UniformVelocityCommandCfg.Ranges = UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-0.0, 0.0), lin_vel_y=(-0.0, 0.0), ang_vel_z=(-0.0, 0.0), heading=(-0.0, 0.0)
    )


@configclass
class GUIKeyboardVelocityCommandStandBitCfg(GUIKeyboardSE3CommandStandBitCfg, UniformVelocityStandBitCommandCfg):
    """Configuration for the velocity command based on keyboard input."""

    class_type: type = GUIKeyboardSE3CommandStandBit

    stand_bit_threshold: float = 0.05

    # Dummy and unused range as UniformVelocityCommand __init__() requires a range
    ranges: UniformVelocityStandBitCommandCfg.Ranges = UniformVelocityStandBitCommandCfg.Ranges(
        lin_vel_x=(-0.0, 0.0), lin_vel_y=(-0.0, 0.0), ang_vel_z=(-0.0, 0.0), heading=(-0.0, 0.0)
    )


@configclass
class GUIKeyboardHeightCommandCfg(GUIKeyboardSE3HeightCommandCfg, HeightCommandCfg):
    """Configuration for the height command based on keyboard input."""

    class_type: type = GUIKeyboardHeightCommand

    # Dummy and unused range as HeightCommand __init__() requires a range
    ranges: HeightCommandCfg.Ranges = HeightCommandCfg.Ranges(height=(0.22, 0.35))


@configclass
class GUIKeyboardWholeBodyCfg(GUIKeyboardSE3CommandCfg, CategoricalCommandCfg):
    """Configuration for the whole-body command based on keyboard input."""

    class_type: type = GUIKeyboardWholeBodyCommand

    nominal_height: float = 0.35

    # -- used for the GUI keyboard command clipping
    ranges = {
        "gui": CategoricalCommandCfg.Ranges(
        lin_vel_x=(-1.0, 1.0),
        lin_vel_y=(-1.0, 1.0),
        ang_vel_z=(-1.0, 1.0),
        height=(0.22, 0.35),
        roll=(-0.35, 0.35),
        pitch=(-0.35, 0.35),
        probability=1.0, # not used
        ),
    }


@configclass
class GUIKeyboardPoseCfg(GUIKeyboardSE3CommandCfg, UniformPoseCommandCfg):
    """Configuration for the SE3 arm end-effector pose based on keyboard input."""

    class_type: type = GUIKeyboardPoseCommand

    pos_sensitivity: float = 0.005
    rot_sensitivity: float = 0.01

    # -- used for the GUI keyboard command clipping
    ranges =UniformPoseCommandCfg.Ranges(
        pos_x=(0.35, 0.5),
        pos_y=(-0.15, 0.15),
        pos_z=(0.15, 0.45),
        roll=(0.0, 0.0),
        pitch=(0.0, 0.0),
        yaw=(0.0, 0.0),
        )