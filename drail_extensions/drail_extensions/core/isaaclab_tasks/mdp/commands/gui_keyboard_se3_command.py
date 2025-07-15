# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for the 2D-pose for locomotion tasks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.managers import CommandTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass

try:
    from isaaclab.devices import Se3Keyboard
except AttributeError:
    print(
        "[ERROR] GUI commands cannot be imported. If you are running this script in a headless mode, this is expected."
    )


class GUIKeyboardSE3Command(CommandTerm):
    r"""Command generator that generates a SE(3) command based on keyboard input."""

    cfg: GUIKeyboardSE3CommandCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: GUIKeyboardSE3CommandCfg, env: ManagerBasedEnv):
        """Initialize the command buffer and the teleop interface.
        Args:
            cfg: The configuration of the command generator.
            env: The environment.
        """
        # initialize the base class
        super().__init__(cfg, env)

        assert cfg.resampling_time_range == (
            0.0,
            0.0,
        ), "Resampling time range for GUIKeyboardCommand must be (0.0, 0.0)"

        if cfg.debug_vis and cfg.asset_name is not None and cfg.body_name is not None:
            # Extract body_idx for visualization
            self.robot: Articulation = env.scene[cfg.asset_name]
            self.body_idx = self.robot.find_bodies(cfg.body_name)[0][0]

        # create buffers to store the command
        self.s3_command = torch.zeros(self.num_envs, 6, device=self.device, dtype=torch.float32)
        self._s3_command_delta = torch.zeros(self.num_envs, 6, device=self.device, dtype=torch.float32)
        # initialize the teleop interface
        self.teleop_interface = Se3Keyboard(pos_sensitivity=cfg.pos_sensitivity, rot_sensitivity=cfg.rot_sensitivity)
        self.teleop_interface.add_callback("O", self._reset_command)
        self.teleop_interface.add_callback("R", self._reset_env)

        print("Teleop interface: ", self.teleop_interface)

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "GUIKeyboardSE3Command:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame.
        Shape is (num_envs, 6) if return_euler_angles is True, otherwise (num_envs, 7)."""

        if self.cfg.return_euler_angles:
            return self.s3_command

        return torch.cat(
            (
                self.s3_command[:, :3],
                math_utils.quat_from_euler_xyz(
                    roll=self.s3_command[:, 3], pitch=self.s3_command[:, 4], yaw=self.s3_command[:, 5]
                ),
            ),
            dim=-1,
        )

    def _resample_command(self, env_ids: Sequence[int]):
        command, _ = self.teleop_interface.advance()
        self._s3_command_delta = torch.from_numpy(command).to(self.device, dtype=torch.float32)
        self.s3_command[env_ids] += self._s3_command_delta

    def _reset_command(self):
        self.s3_command.zero_()

    def _reset_env(self):
        self._env.reset()

    def _update_metrics(self):
        pass

    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                # -- goal pose
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
                # -- current body pose
                self.current_pose_visualizer = VisualizationMarkers(self.cfg.current_pose_visualizer_cfg)
            # set their visibility to true
            self.goal_pose_visualizer.set_visibility(True)
            self.current_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
                self.current_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return

        # update the markers
        # -- goal pose
        s3_command_quat = math_utils.quat_from_euler_xyz(
            roll=self.s3_command[:, 3], pitch=self.s3_command[:, 4], yaw=self.s3_command[:, 5]
        )

        self.goal_pose_visualizer.visualize(self.s3_command[:, :3], s3_command_quat)
        # -- current body pose
        body_link_state_w = self.robot.data.body_state_w[:, self.body_idx]
        self.current_pose_visualizer.visualize(body_link_state_w[:, :3], body_link_state_w[:, 3:7])


@configclass
class GUIKeyboardSE3CommandCfg(CommandTermCfg):
    """Configuration for the SE(3) command based on keyboard input."""

    class_type: type = GUIKeyboardSE3Command

    asset_name: str | None = None
    """Name of the asset for visualizing the command."""

    body_name: str | None = None
    """Name of the body for visualizing the command."""

    pos_sensitivity: float = 0.05
    """The sensitivity for the position command. Defaults to 0.05."""

    rot_sensitivity: float = 0.05
    """The sensitivity for the rotation command. Defaults to 0.05."""

    return_euler_angles: bool = True
    """Whether to return the euler angles or the quaternion. Defaults to True."""

    @configclass
    class OffsetCfg:
        """Offset of the command."""

        # root position
        pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Position of the root in simulation world frame. Defaults to (0.0, 0.0, 0.0)."""
        rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
        """Quaternion rotation (w, x, y, z) of the root in simulation world frame.
        Defaults to (1.0, 0.0, 0.0, 0.0).
        """

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/goal_pose")
    """The configuration for the goal pose visualization marker. Defaults to FRAME_MARKER_CFG."""

    current_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/body_pose"
    )
    """The configuration for the current pose visualization marker. Defaults to FRAME_MARKER_CFG."""

    # Set the scale of the visualization markers to (0.1, 0.1, 0.1)
    goal_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    current_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)

    resampling_time_range: tuple[float, float] = (0.0, 0.0)
    """The time range for the resampling of the command. Defaults to (0.0, 0.0). This quantity must be set to
    (0.0, 0.0) to listen to the keyboard input at every time step. To control sensitivity,
    use the :attr:`pos_sensitivity` and :attr:`rot_sensitivity` parameters."""
