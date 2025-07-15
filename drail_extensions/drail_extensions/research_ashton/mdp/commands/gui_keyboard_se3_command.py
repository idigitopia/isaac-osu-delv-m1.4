# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for the 2D-pose for locomotion tasks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.managers import CommandTerm
from isaaclab.assets import Articulation
from isaaclab.markers import VisualizationMarkers
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from .gui_keyboard_se3_command_cfg import (
        GUIKeyboardSE3CommandCfg,
        GUIKeyboardSE3CommandStandBitCfg,
        )

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

        # create buffers to store the command
        self.s3_command = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float32)
        self._s3_command_delta = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float32)
        # initialize the teleop interface
        self.teleop_interface = Se3Keyboard(pos_sensitivity=cfg.pos_sensitivity, rot_sensitivity=cfg.rot_sensitivity)
        self.teleop_interface.add_callback("KEY_0", self._reset_command)
        self.teleop_interface.add_callback("NUMPAD_0", self._reset_command)
        self.teleop_interface.add_callback("R", self._reset_command)

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "GUIKeyboardSE3Command:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, 3)."""
        return self.s3_command

    def _resample_command(self, env_ids: Sequence[int]):
        command, _ = self.teleop_interface.advance()
        self._s3_command_delta = torch.from_numpy(command[:3]).to(self.device, dtype=torch.float32)
        self.s3_command[env_ids] += self._s3_command_delta

    def _reset_command(self):
        self.s3_command.zero_()

    def _update_metrics(self):
        pass

    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass

    def _debug_vis_callback(self, event):
        pass


class GUIKeyboardSE3CommandStandBit(CommandTerm):
    r"""Command generator that generates a SE(3) command with a stand bit based on keyboard input."""

    cfg: GUIKeyboardSE3CommandStandBitCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: GUIKeyboardSE3CommandStandBitCfg, env: ManagerBasedEnv):
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

        # create buffers to store the command
        self.s3_command = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float32)
        self.stand_bit = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self._s3_command_delta = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float32)
        # initialize the teleop interface
        self.teleop_interface = Se3Keyboard(pos_sensitivity=cfg.pos_sensitivity, rot_sensitivity=cfg.rot_sensitivity)
        self.teleop_interface.add_callback("KEY_0", self._reset_command)
        self.teleop_interface.add_callback("NUMPAD_0", self._reset_command)
        self.teleop_interface.add_callback("R", self._reset_command)

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "GUIKeyboardSE3Command:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame with a stand bit. Shape is (num_envs, 4)."""
        return torch.cat((self.s3_command, self.stand_bit.view(-1, 1)), dim=1)

    def _resample_command(self, env_ids: Sequence[int]):
        command, _ = self.teleop_interface.advance()
        self._s3_command_delta = torch.from_numpy(command[:3]).to(self.device, dtype=torch.float32)
        self.s3_command[env_ids] += self._s3_command_delta

        # Update the stand bit
        self.stand_bit[env_ids] = torch.where(
            torch.norm(self.s3_command[env_ids], dim=1) < self.cfg.stand_bit_threshold,
            1.0,  # Standing env
            0.0,  # Walking env
        ).to(self.device)

    def _reset_command(self):
        self.s3_command.zero_()
        self.stand_bit[:] = 1.0

    def _update_metrics(self):
        pass

    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass

    def _debug_vis_callback(self, event):
        pass


class GUIKeyboardHeightCommand(CommandTerm):
    r"""Command generator that generates a height command based on keyboard input."""

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

        # create buffers to store the command
        self.height_command = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.float32)
        self.height_command[:, 0] = cfg.nominal_height
        self._height_command_delta = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.float32)
        # initialize the teleop interface
        self.teleop_interface = Se3Keyboard(pos_sensitivity=cfg.pos_sensitivity, rot_sensitivity=cfg.rot_sensitivity)
        self.teleop_interface.add_callback("KEY_0", self._reset_command)
        self.teleop_interface.add_callback("NUMPAD_0", self._reset_command)
        self.teleop_interface.add_callback("R", self._reset_command)

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "GUIKeyboardSE3Command:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """The desired base height command in the base frame. Shape is (num_envs)."""
        return self.height_command

    def _resample_command(self, env_ids: Sequence[int]):
        command, _ = self.teleop_interface.advance()
        # self._height_command_delta = torch.from_numpy(command[3]).to(self.device, dtype=torch.float32)
        self._height_command_delta = torch.tensor(command[3], dtype=torch.float32, device=self.device).expand(len(env_ids), 1)
        self.height_command[env_ids] += self._height_command_delta
        # Clip to the specified range
        self.height_command[env_ids] = torch.clamp(
            self.height_command[env_ids],
            min=self.cfg.ranges.height[0],
            max=self.cfg.ranges.height[1],
        )
        print(f"Height command: {self.height_command[env_ids]}", end="\r")

    def _reset_command(self):
        self.height_command[:] = self.cfg.nominal_height

    def _update_metrics(self):
        pass

    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass

    def _debug_vis_callback(self, event):
        pass


class GUIKeyboardSE3WholeBodyCommand(CommandTerm):
    r"""Command generator that generates a whole-body command based on keyboard input."""

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

        # just velocity command flag
        self._just_vel_cmd = cfg.just_vel_cmd

        # extract ranges
        self._gui_ranges = self.cfg.ranges["gui"]

        # create buffers to store the command
        self.wholebody_command = torch.zeros(self.num_envs, 7, device=self.device, dtype=torch.float32)
        self.wholebody_command[:, 4] = cfg.nominal_height
        self._wholebody_command_delta = torch.zeros(self.num_envs, 6, device=self.device, dtype=torch.float32)
        # initialize the teleop interface
        self.teleop_interface = Se3Keyboard(pos_sensitivity=cfg.pos_sensitivity, rot_sensitivity=cfg.rot_sensitivity)
        self.teleop_interface.add_callback("KEY_0", self._reset_command)
        self.teleop_interface.add_callback("NUMPAD_0", self._reset_command)
        self.teleop_interface.add_callback("R", self._reset_command)

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "GUIKeyboardSE3Command:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """The desired whole body command in the base frame. Shape is (num_envs, 7)."""
        if self._just_vel_cmd:
            return self.wholebody_command[:, :4]
        return self.wholebody_command

    def _resample_command(self, env_ids: Sequence[int]):
        command, _ = self.teleop_interface.advance()
        self._wholebody_command_delta = torch.from_numpy(command).to(self.device, dtype=torch.float32)
        self.wholebody_command[env_ids, :3] += self._wholebody_command_delta[:3]
        self.wholebody_command[env_ids, 4:] += self._wholebody_command_delta[3:6]

        # Clip all commands to their specified ranges
        # -- lin vel x
        self.wholebody_command[env_ids, 0] = torch.clamp(
            self.wholebody_command[env_ids, 0],
            min=self._gui_ranges.lin_vel_x[0],
            max=self._gui_ranges.lin_vel_x[1],
        )
        # -- lin vel y
        self.wholebody_command[env_ids, 1] = torch.clamp(
            self.wholebody_command[env_ids, 1],
            min=self._gui_ranges.lin_vel_y[0],
            max=self._gui_ranges.lin_vel_y[1],
        )
        # -- ang vel z
        self.wholebody_command[env_ids, 2] = torch.clamp(
            self.wholebody_command[env_ids, 2],
            min=self._gui_ranges.ang_vel_z[0],
            max=self._gui_ranges.ang_vel_z[1],
        )
        # -- height
        self.wholebody_command[env_ids, 4] = torch.clamp(
            self.wholebody_command[env_ids, 4],
            min=self._gui_ranges.height[0],
            max=self._gui_ranges.height[1],
        )
        # -- roll
        self.wholebody_command[env_ids, 5] = torch.clamp(
            self.wholebody_command[env_ids, 5],
            min=self._gui_ranges.roll[0],
            max=self._gui_ranges.roll[1],
        )
        # -- pitch
        self.wholebody_command[env_ids, 6] = torch.clamp(
            self.wholebody_command[env_ids, 6],
            min=self._gui_ranges.pitch[0],
            max=self._gui_ranges.pitch[1],
        )
        # --set the stand bit
        self.wholebody_command[env_ids, 3] = torch.where(
            torch.norm(self.wholebody_command[env_ids, :3], dim=1) < 0.05,
            1.0,  # Standing env
            0.0,  # Walking env
        )

        if self._just_vel_cmd:
            print(f"Velocity command: {self.wholebody_command[env_ids, :4]}", end="\r")
        else:
            print(f"Whole body command: {self.wholebody_command[env_ids]}", end="\r")


    def _reset_command(self):
        self.wholebody_command.zero_()
        self.wholebody_command[:, 4] = self.cfg.nominal_height
        self.wholebody_command[:, 3] = 1.0

    def _update_metrics(self):
        pass

    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass

    def _debug_vis_callback(self, event):
        pass


class GUIKeyboardSE3PoseCommand(CommandTerm):
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
        self.s3_command = torch.zeros(self.num_envs, 7, device=self.device, dtype=torch.float32)
        self.pose_command_w = torch.zeros_like(self.s3_command)
        self._s3_command_delta = torch.zeros(self.num_envs, 6, device=self.device, dtype=torch.float32)
        self._start_pose = torch.tensor([0.3,  0.07,  0.4, 1.0,  0.0, 0.0, 0.0], device=self.device, dtype=torch.float32)
        self.s3_command = self._start_pose.expand(self.num_envs, 7).clone()

        # initialize the teleop interface
        self.teleop_interface = Se3Keyboard(pos_sensitivity=cfg.pos_sensitivity, rot_sensitivity=cfg.rot_sensitivity)
        self.teleop_interface.add_callback("KEY_0", self._reset_command)
        self.teleop_interface.add_callback("NUMPAD_0", self._reset_command)
        self.teleop_interface.add_callback("R", self._reset_env)

        print("Teleop interface: ", self.teleop_interface)

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "GUIKeyboardSE3Command:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """The desired command (num_envs, 7)."""
        return self.s3_command

    def _resample_command(self, env_ids: Sequence[int]):
        command, _ = self.teleop_interface.advance()
        self._s3_command_delta = torch.from_numpy(command).to(self.device, dtype=torch.float32)
        # Position add as is
        self.s3_command[env_ids, :3] += self._s3_command_delta[:3]

        # get delta rotation
        delta_quat = math_utils.quat_from_euler_xyz(
            self._s3_command_delta[3], self._s3_command_delta[4], self._s3_command_delta[5]
        )
        delta_quat = delta_quat.expand(self.num_envs, 4)
        # rotate the current command by the delta rotation
        self.s3_command[env_ids, 3:] = math_utils.quat_mul(
            self.s3_command[env_ids, 3:], delta_quat
        )
        # unique
        if self.cfg.make_quat_unique:
            self.s3_command[env_ids, 3:] = math_utils.quat_unique(self.s3_command[env_ids, 3:])

        # clip position
        self.s3_command[env_ids, 0] = torch.clamp(
            self.s3_command[env_ids, 0],
            min=self.cfg.ranges.pos_x[0],
            max=self.cfg.ranges.pos_x[1],
        )
        self.s3_command[env_ids, 1] = torch.clamp(
            self.s3_command[env_ids, 1],
            min=self.cfg.ranges.pos_y[0],
            max=self.cfg.ranges.pos_y[1],
        )
        self.s3_command[env_ids, 2] = torch.clamp(
            self.s3_command[env_ids, 2],
            min=self.cfg.ranges.pos_z[0],
            max=self.cfg.ranges.pos_z[1],
        )

        # # update the world pose command for visualization
        self.pose_command_w[:, :3], self.pose_command_w[:, 3:] = math_utils.combine_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            self.s3_command[:, :3],
            self.s3_command[:, 3:],
        )
        # self.pose_command_w = self.s3_command.clone()

        print(f"Pose command: {self.s3_command[env_ids]}", end="\r")

    def _reset_command(self):
        self.s3_command = self._start_pose.expand(self.num_envs, 7).clone()

    def _reset_env(self):
        self._env.reset()

    def _update_metrics(self):
        pass

    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass

    def _debug_vis_callback(self, event):
        pass
