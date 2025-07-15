# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing whole body command generator for locomotion tasks."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers

from isaaclab.envs.mdp.commands import UniformVelocityCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .whole_body_command_cfg import *


class UniformVelocityStandBitCommand(UniformVelocityCommand):
    r"""Adds a stand bit to the uniform velocity command."""

    cfg: UniformVelocityStandBitCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: UniformVelocityStandBitCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment.

        """

        # initialize the base class
        super().__init__(cfg, env)

        self._stand_threshold = cfg.stand_bit_threshold
        self.stand_bit = torch.zeros(self.num_envs, device=self.device)

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command with stand bit."""

        # Concatenate vel_command_b (num_envs x 3) with the stand bit (num_envs x 1)
        # to create a command of shape (num_envs x 4)
        return torch.cat((self.vel_command_b, self.stand_bit.view(-1,1)), dim=1)

    def _update_command(self):
        # Update the command from the base class
        super()._update_command()

        # Set the stand bit to 1 if the velocity command is below the threshold
        self.stand_bit = torch.where(
            torch.norm(self.vel_command_b, dim=1) < self._stand_threshold,
            1.0, # Standing env
            0.0, # Walking env
        ).to(self.device)


class HeightCommand(CommandTerm):
    r"""Command generator for height control."""

    cfg: HeightCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: HeightCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment.

        """
        # initialize the base class
        super().__init__(cfg, env)

        # obtain the robot asset
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.body_idx = self.robot.find_bodies(cfg.body_name)[0][0]

        # if sensor name is provided obtain the sensor asset
        if cfg.sensor_name is not None:
            self.sensor = env.scene[cfg.sensor_name]
        else:
            self.sensor = None

        # create buffers to store the command
        # -- commands: height
        self.height_command_b = torch.zeros(self.num_envs, 1, device=self.device)
        self.is_nominal_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # -- metrics
        self.metrics["error_height"] = torch.zeros(self.num_envs, device=self.device)

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired height command. Shape is (num_envs)."""
        return self.height_command_b

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # time for which the command was executed
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt

        # target height
        target_height = self.height_command_b[:, 0]
        if self.sensor is not None:
            # adjust the target height based on the sensor data for rough terrain
            target_height += torch.mean(self.sensor.data.ray_hits_w[..., 2], dim=1)

        # logs data
        self.metrics["error_height"] += (
            torch.abs(target_height - self.robot.data.body_pos_w[:, self.body_idx, 2]) / max_command_step
        )

    def _resample_command(self, env_ids: Sequence[int]):
        # sample the height command
        r = torch.empty(len(env_ids), device=self.device)
        self.height_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.height)
        # update nominal height envs
        self.is_nominal_env[env_ids] = r.uniform_(0.0, 1.0) < self.cfg.rel_nominal_envs

    def _update_command(self):
        """Post process the height command."""

        # Enforce nominal height for nominal environments
        nominal_env_ids = self.is_nominal_env.nonzero(as_tuple=False).flatten()
        self.height_command_b[nominal_env_ids, 0] = self.cfg.nominal_height


class BodyTiltCommand(CommandTerm):
    r"""Command generator for body tilt control for the root of the asset."""

    cfg: BodyTiltCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: BodyTiltCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment.

        """
        # initialize the base class
        super().__init__(cfg, env)

        # obtain the robot asset
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.body_idx = self.robot.find_bodies(cfg.body_name)[0][0]

        # create buffers to store the command
        # -- commands: roll, pitch in root frame
        self.tilt_command_b = torch.zeros(self.num_envs, 2, device=self.device)
        self.tilt_command_vec = torch.zeros(self.num_envs, 3, device=self.device)
        self.is_nominal_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # -- metrics
        self.metrics["error_roll"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_pitch"] = torch.zeros(self.num_envs, device=self.device)

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired body tilt command. Shape is (num_envs, 2)."""
        return self.tilt_command_b

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # TODO: Double check this
        # time for which the command was executed
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt

        # get the current body orientation
        body_quat = self.robot.data.body_quat_w[:, self.body_idx, :].squeeze(1)
        # get euler angles from the quaternion
        roll, pitch, _ = math_utils.euler_xyz_from_quat(body_quat)
        # wrap to pi
        roll = math_utils.wrap_to_pi(roll)
        pitch = math_utils.wrap_to_pi(pitch)

        # logs data
        self.metrics["error_roll"] += torch.abs(self.tilt_command_b[:, 0] - roll) / max_command_step
        self.metrics["error_pitch"] += torch.abs(self.tilt_command_b[:, 1] - pitch) / max_command_step

    def _resample_command(self, env_ids: Sequence[int]):
        # sample the tilt command
        r = torch.empty(len(env_ids), device=self.device)
        self.tilt_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.roll)
        self.tilt_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.pitch)

        # update nominal tilt envs
        self.is_nominal_env[env_ids] = r.uniform_(0.0, 1.0) < self.cfg.rel_zero_tilt_envs

    def _update_command(self):
        """Post process the tilt command."""

        # Enforce nominal tilt for nominal environments
        nominal_env_ids = self.is_nominal_env.nonzero(as_tuple=False).flatten()
        self.tilt_command_b[nominal_env_ids, :] = 0.0
        self.tilt_command_vec[:, :2] = self.tilt_command_b

    """
    Visualization functions.
    """

    def _set_debug_vis_impl(self, debug_vis):
        # create markers if necessary for the first time
        if debug_vis:
            if not hasattr(self, "goal_tile_visualizer"):
                # -- goal tilt
                self.goal_tilt_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
                # -- current body tilt
                self.current_tilt_visualizer = VisualizationMarkers(self.cfg.current_pose_visualizer_cfg)
            # set their visibility to true
            self.goal_tilt_visualizer.set_visibility(True)
            self.current_tilt_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_tilt_visualizer"):
                self.goal_tilt_visualizer.set_visibility(False)
                self.current_tilt_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # get the current body pose
        body_pose = self.robot.data.body_state_w[:, self.body_idx, :]
        # update the markers
        # -- current body tilt
        self.current_tilt_visualizer.visualize(body_pose[:, :3], body_pose[:, 3:7])
        # -- goal tilt
        # self.tilt_command_vec[:, :2] = self.tilt_command_b
        self.tilt_command_vec = math_utils.quat_apply_yaw(body_pose[:, 3:7], self.tilt_command_vec)
        goal_quat = math_utils.quat_from_euler_xyz(self.tilt_command_vec[:, 0], self.tilt_command_vec[:, 1], self.tilt_command_vec[:, 2])
        self.goal_tilt_visualizer.visualize(body_pose[:, :3], goal_quat)


class CategoricalCommand(CommandTerm):
    r"""Command generator that generates a velocity command in SE(2) from categorical distribution of ranges. With height and body tile commands.

    The command comprises of a linear velocity in x and y direction and an angular velocity around
    the z-axis, stand bit, base height, roll, and pitch. It is given in the robot's base frame.
    """

    cfg: CategoricalCommandCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: CategoricalCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.

        Raises:
            ValueError: If the heading command is active but the heading range is not provided.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # obtain the robot asset
        # -- robot
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.body_idx = self.robot.find_bodies(cfg.body_name)[0][0]

        # only velocity commands flag
        self._just_vel_cmd = cfg.just_vel_cmd

        # crete buffers to store the command
        # -- command: x vel, y vel, yaw vel, stand bit, height, roll, pitch
        self.wholebody_command_b = torch.zeros(self.num_envs, 7, device=self.device)
        # -- metrics
        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)
        if not self._just_vel_cmd:
            self.metrics["error_height"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics["error_roll"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics["error_pitch"] = torch.zeros(self.num_envs, device=self.device)

        # normalize the ranges probability
        self.ranges = list(self.cfg.ranges.values())
        self.ranges_names = list(self.cfg.ranges.keys())
        self.ranges_prob = torch.tensor([range.probability for range in self.ranges], device=self.device)
        # normalize the ranges probability
        self.ranges_prob = self.ranges_prob / self.ranges_prob.sum()

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "CategoricalCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tRanges category: {zip(self.ranges_names, self.ranges_prob)}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, 7)."""
        if self._just_vel_cmd:
            # return only the velocity command shape (num_envs, 4)
            return self.wholebody_command_b[:, :4]
        # return the whole command shape (num_envs, 7)
        return self.wholebody_command_b

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # time for which the command was executed
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        # logs data
        self.metrics["error_vel_xy"] += (
            torch.norm(self.wholebody_command_b[:, :2] - self.robot.data.root_lin_vel_b[:, :2], dim=-1) / max_command_step
        )
        self.metrics["error_vel_yaw"] += (
            torch.abs(self.wholebody_command_b[:, 2] - self.robot.data.root_ang_vel_b[:, 2]) / max_command_step
        )
        if not self._just_vel_cmd:
            self.metrics["error_height"] += (
                torch.abs(self.wholebody_command_b[:, 4] - self.robot.data.body_pos_w[:, self.body_idx, 2]) / max_command_step
            )

            # get the current body orientation
            body_quat = self.robot.data.body_quat_w[:, self.body_idx, :].squeeze(1)
            # get euler angles from the quaternion
            roll, pitch, _ = math_utils.euler_xyz_from_quat(body_quat)
            # wrap to pi
            roll = math_utils.wrap_to_pi(roll)
            pitch = math_utils.wrap_to_pi(pitch)

            self.metrics["error_roll"] += torch.abs(self.wholebody_command_b[:, 5] - roll) / max_command_step
            self.metrics["error_pitch"] += torch.abs(self.wholebody_command_b[:, 6] - pitch) / max_command_step

    def _resample_command(self, env_ids: Sequence[int]):
        # sample the command from the categorical distribution
        idx = torch.multinomial(self.ranges_prob, num_samples=1).item()

        r = torch.empty(len(env_ids), device=self.device)
        # -- linear velocity - x direction
        self.wholebody_command_b[env_ids, 0] = r.uniform_(*self.ranges[idx].lin_vel_x)
        # -- linear velocity - y direction
        self.wholebody_command_b[env_ids, 1] = r.uniform_(*self.ranges[idx].lin_vel_y)
        # -- ang vel yaw - rotation around z
        self.wholebody_command_b[env_ids, 2] = r.uniform_(*self.ranges[idx].ang_vel_z)

        if not self._just_vel_cmd:
            # -- height
            self.wholebody_command_b[env_ids, 4] = r.uniform_(*self.ranges[idx].height)
            # -- roll
            self.wholebody_command_b[env_ids, 5] = r.uniform_(*self.ranges[idx].roll)
            # -- pitch
            self.wholebody_command_b[env_ids, 6] = r.uniform_(*self.ranges[idx].pitch)

        # -- set the stand bit
        self.wholebody_command_b[env_ids, 3] = torch.where(
            torch.norm(self.wholebody_command_b[env_ids, :3], dim=1) < 0.01,
            1.0,
            0.0
        )

    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first tome
            if not hasattr(self, "goal_vel_visualizer"):
                # -- goal
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                # -- current
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            # set their visibility to true
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # get marker location
        # -- base state
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        # -- resolve the scales and quaternions
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.command[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        # display markers
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    """
    Internal helpers.
    """

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        # obtain default scale of the marker
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        # arrow-direction
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        # convert everything back from base to world frame
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat
