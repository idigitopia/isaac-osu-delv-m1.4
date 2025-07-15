# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG, BLUE_ARROW_X_MARKER_CFG

from isaaclab.envs.mdp.commands import UniformVelocityCommandCfg

from .whole_body_command import (
    UniformVelocityStandBitCommand,
    HeightCommand,
    BodyTiltCommand,
    CategoricalCommand,
)


@configclass
class UniformVelocityStandBitCommandCfg(UniformVelocityCommandCfg):
    """Configuration for the uniform velocity with a stand bit command generator."""
    class_type: type = UniformVelocityStandBitCommand

    stand_bit_threshold: float = 0.05
    """Threshold for the stand bit. Defaults to 0.05.

    The stand bit is a 0/1 value set when the sampled velocity command is below this threshold.
    """


@configclass
class HeightCommandCfg(CommandTermCfg):
    """Configuration for the height command generator for a body of the asset."""

    class_type: type = HeightCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    body_name: str = MISSING
    """Name of the body in the asset for which the commands are generated."""

    sensor_name: str | None = None
    """Name of the height sensor in the environment when using rough terrain."""

    nominal_height: float = 0.0
    """Nominal height of the asset."""

    rel_nominal_envs: float = 0.0
    """The sampled probability of environments that should be at the nominal height. Defaults to 0.0."""

    @configclass
    class Ranges:
        """Configuration for the height command ranges."""

        height: tuple[float, float] = MISSING
        """Range of the height command (in m)."""

    ranges: Ranges = MISSING
    """Ranges for the height command."""


@configclass
class BodyTiltCommandCfg(CommandTermCfg):
    """Configuration for the body tilt command generator for a body of the asset."""

    class_type: type = BodyTiltCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    body_name: str = MISSING
    """Name of the body in the asset for which the commands are generated."""

    rel_zero_tilt_envs: float = 0.0
    """The sampled probability of environments that should be at the zero tilt. Defaults to 0.0."""

    @configclass
    class Ranges:
        """Configuration for the body tilt command ranges."""

        roll: tuple[float, float] = MISSING
        """Range of the roll command (in rad)."""

        pitch: tuple[float, float] = MISSING
        """Range of the pitch command (in rad)."""

    ranges: Ranges = MISSING
    """Ranges for the body tilt command."""

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/goal_pose")
    """The configuration for the goal pose visualization marker. Defaults to FRAME_MARKER_CFG."""

    current_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/body_pose"
    )
    """The configuration for the current pose visualization marker. Defaults to FRAME_MARKER_CFG."""

    # Set the scale of the visualization markers to (0.1, 0.1, 0.1)
    goal_pose_visualizer_cfg.markers["frame"].scale = (0.15, 0.15, 0.15)
    current_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)


@configclass
class CategoricalCommandCfg(CommandTermCfg):
    """Configuration for the categorical command generator."""

    class_type: type = CategoricalCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    body_name: str = MISSING
    """Name of the body in the asset for which the commands are generated."""

    just_vel_cmd: bool = False
    """Whether to only use the velocity command. Defaults to False.
    If True, the command will only be velocity commands and stand bit.
    Defaults to whole-body commands."""

    @configclass
    class Ranges:
        """Uniform distribution ranges for the velocity commands."""

        lin_vel_x: tuple[float, float] = MISSING
        """Range for the linear-x velocity command (in m/s)."""

        lin_vel_y: tuple[float, float] = MISSING
        """Range for the linear-y velocity command (in m/s)."""

        ang_vel_z: tuple[float, float] = MISSING
        """Range for the angular-z velocity command (in rad/s)."""

        height: tuple[float, float] = MISSING
        """Range for the height command (in m)."""

        roll: tuple[float, float] = MISSING
        """Range for the roll command (in rad)."""

        pitch: tuple[float, float] = MISSING
        """Range for the pitch command (in rad)."""

        probability: tuple[float, float] = MISSING
        """Range for the probability of the command (in [0, 1])."""

    ranges: dict[str, Ranges] = MISSING
    """Distribution ranges for the velocity commands."""

    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    """The configuration for the goal velocity visualization marker. Defaults to GREEN_ARROW_X_MARKER_CFG."""

    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )
    """The configuration for the current velocity visualization marker. Defaults to BLUE_ARROW_X_MARKER_CFG."""

    # Set the scale of the visualization markers to (0.5, 0.5, 0.5)
    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)