# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

import isaaclab.utils.math as math_utils
import numpy as np
from isaaclab.managers import ManagerTermBase
from isaaclab.managers.manager_term_cfg import RewardTermCfg
from isaaclab.sensors import ContactSensor


def base_height_out_of_range(
    env: ManagerBasedRLEnv,
    min_height: float = -np.inf,
    max_height: float = np.inf,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = None,
) -> torch.Tensor:
    """Penalize asset height from its target using exponential kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    curr_height = asset.data.root_link_pos_w[:, 2]

    out_of_range = (curr_height < min_height) | (curr_height > max_height)

    return out_of_range  # noqa: R504


class style_reward(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        # initialize the base class
        super().__init__(cfg, env)

        # Initialize the computed values to store the probability and reward
        self.computed_values = {}

    def compute_logits(self, amp_obs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("This function must be set in order to compute the style reward")

    def __call__(self, env: ManagerBasedRLEnv, amp_observation_key: str, reward_scale: float = 2.0) -> torch.Tensor:
        with torch.no_grad():
            # Get the amp observation
            amp_obs = env.observation_manager._obs_buffer[amp_observation_key]

            # Compute the discriminator logits and probability
            disc_logits = self.compute_logits(amp_obs)
            prob = torch.sigmoid(disc_logits).squeeze(-1)

            # Compute the reward
            reward = -torch.log(torch.maximum(1 - prob, torch.tensor(0.0001, device=self.device)))
            reward *= reward_scale

            # Store computed values
            self.computed_values["prob"] = prob
            self.computed_values["reward"] = reward

            # Return the style reward
            return reward


def position_command_error_tanh(env: ManagerBasedRLEnv, std: float, command_name: str) -> torch.Tensor:
    """Reward position tracking with tanh kernel."""
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    distance = torch.norm(des_pos_b, dim=1)
    return 1 - torch.tanh(distance / std)


def heading_command_error_abs(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Penalize tracking orientation error."""
    command = env.command_manager.get_command(command_name)
    heading_b = command[:, 3]
    return heading_b.abs()


def feet_air_time_v2(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """
    Modification to isaaclab_tasks.manager_based.locomotion.velocity.mdp.feet_air_time
    to prevent rewarding for body velocity at zero angular velocity command.

    Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    cmd = env.command_manager.get_command(command_name)
    return reward * (cmd.norm(dim=1) > 0.0)


class GaitReward_v2(ManagerTermBase):
    """
    Modification to isaaclab_tasks.manager_based.locomotion.velocity.config.spot.mdp.GaitReward
    to prevent rewarding for body velocity at zero command.

    Gait enforcing reward term for quadrupeds.

    This reward penalizes contact timing differences between selected foot pairs defined in
    :attr:`synced_feet_pair_names`
    to bias the policy towards a desired gait, i.e trotting, bounding, or pacing. Note that this reward is only for
    quadrupedal gaits with two pairs of synchronized feet.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.std: float = cfg.params["std"]
        self.max_err: float = cfg.params["max_err"]
        # self.velocity_threshold: float = cfg.params["velocity_threshold"]
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        # match foot body names with corresponding foot body ids
        synced_feet_pair_names = cfg.params["synced_feet_pair_names"]
        if (
            len(synced_feet_pair_names) != 2
            or len(synced_feet_pair_names[0]) != 2
            or len(synced_feet_pair_names[1]) != 2
        ):
            raise ValueError("This reward only supports gaits with two pairs of synchronized feet, like trotting.")
        synced_feet_pair_0 = self.contact_sensor.find_bodies(synced_feet_pair_names[0])[0]
        synced_feet_pair_1 = self.contact_sensor.find_bodies(synced_feet_pair_names[1])[0]
        self.synced_feet_pairs = [synced_feet_pair_0, synced_feet_pair_1]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        max_err: float,
        # velocity_threshold: float,
        synced_feet_pair_names,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        command_name: str,
    ) -> torch.Tensor:
        """Compute the reward.

        This reward is defined as a multiplication between six terms where two of them enforce pair feet
        being in sync and the other four rewards if all the other remaining pairs are out of sync

        Args:
            env: The RL environment instance.
        Returns:
            The reward value.
        """
        # for synchronous feet, the contact (air) times of two feet should match
        sync_reward_0 = self._sync_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[0][1])
        sync_reward_1 = self._sync_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[1][1])
        sync_reward = sync_reward_0 * sync_reward_1
        # for asynchronous feet, the contact time of one foot should match the air time of the other one
        async_reward_0 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][0])
        async_reward_1 = self._async_reward_func(self.synced_feet_pairs[0][1], self.synced_feet_pairs[1][1])
        async_reward_2 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][1])
        async_reward_3 = self._async_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[0][1])
        async_reward = async_reward_0 * async_reward_1 * async_reward_2 * async_reward_3
        # only enforce gait if cmd > 0
        cmd = env.command_manager.get_command(command_name)
        return sync_reward * async_reward * (cmd.norm(dim=1) > 0.0)

    """
    Helper functions.
    """

    def _sync_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between the most recent air time and contact time of synced feet pairs.
        se_air = torch.clip(torch.square(air_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        se_contact = torch.clip(torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_air + se_contact) / self.std)

    def _async_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward anti-synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between opposing contact modes air time of feet 1 to contact time of feet 2
        # and contact time of feet 1 to air time of feet 2) of feet pairs that are not in sync with each other.
        se_act_0 = torch.clip(torch.square(air_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        se_act_1 = torch.clip(torch.square(contact_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_act_0 + se_act_1) / self.std)


def foot_clearance_reward_v2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    std: float,
    tanh_mult: float,
    command_name: str,
) -> torch.Tensor:
    """
    Modification to isaaclab_tasks.manager_based.locomotion.velocity.config.spot.mdp.foot_clearance_reward
    to only reward when the command is non-zero.

    Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    cmd = env.command_manager.get_command(command_name)
    # Reward only when the command is non-zero
    return torch.exp(-torch.sum(reward, dim=1) / std) * (torch.norm(cmd, dim=1) > 0.0)


def feet_contact_quadruped(env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward feet contact when the command is zero"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # If foot is in contact in current step
    contact = (contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids].norm(dim=-1) >= 0.1).any(dim=1)

    # Reward for exactly 4 feet in contact
    reward = contact.sum(dim=-1) == 4

    cmd = env.command_manager.get_command(command_name)

    # Reward only when the command is zero
    return reward * (cmd.norm(dim=1) == 0)


def air_time_variance_penalty_v2(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, command_name: str) -> torch.Tensor:
    """
    Modification to isaaclab_tasks.manager_based.locomotion.velocity.config.spot.mdp.air_time_variance_penalty
    to only reward when the command is non-zero.

    Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    cmd = env.command_manager.get_command(command_name)
    # Reward only when the command is non-zero
    return (
        torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(torch.clip(last_contact_time, max=0.5), dim=1)
    ) * (cmd.norm(dim=1) > 0.0)


def base_linear_velocity_reward_v2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    std: float,
    ramp_at_vel: float = 1.0,
    ramp_rate: float = 0.5,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using abs exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    target = env.command_manager.get_command(command_name)[:, :2]
    lin_vel_error = torch.linalg.norm((target - asset.data.root_lin_vel_b[:, :2]), dim=1)
    # fixed 1.0 multiple for tracking below the ramp_at_vel value, then scale by the rate above
    vel_cmd_magnitude = torch.linalg.norm(target, dim=1)
    velocity_scaling_multiple = torch.clamp(1.0 + ramp_rate * (vel_cmd_magnitude - ramp_at_vel), min=1.0)
    return torch.exp(-lin_vel_error / std) * velocity_scaling_multiple


def base_angular_velocity_reward_v2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, std: float, command_name: str = "base_velocity"
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using abs exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    target = env.command_manager.get_command(command_name)[:, 2]
    ang_vel_error = torch.linalg.norm((target - asset.data.root_ang_vel_b[:, 2]).unsqueeze(1), dim=1)
    return torch.exp(-ang_vel_error / std)


def joint_position_penalty_v2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float,
    velocity_threshold: float,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    reward = torch.linalg.norm((asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    return torch.where(torch.logical_or(cmd > 0.0, body_vel > velocity_threshold), reward, stand_still_scale * reward)


def action_smoothness_penalty_v2(env: ManagerBasedRLEnv, action_name: str | None = None) -> torch.Tensor:
    """Penalize large instantaneous changes in the network action output"""
    if action_name is None:
        return torch.linalg.norm((env.action_manager.action - env.action_manager.prev_action), dim=1)
    return torch.linalg.norm(
        (
            env.action_manager.get_term(action_name).raw_actions
            - env.action_manager.get_term(action_name).previous_raw_actions
        ),
        dim=1,
    )


class pos_rel_reward_tanh(ManagerTermBase):
    """Reward based on the relative position of target frames with respect to source frame in frame_cfg using tanh
    kernel. If multiple target frames are provided, the reward is the mean of the rewards for each target frame."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self._frame = env.scene[cfg.params["frame_cfg"].name]
        self._axes = torch.tensor(["xyz".index(ax) for ax in cfg.params.get("axes", "xyz")], device=self.device).long()
        self._target_frame_idx = self._frame._target_frame_names.index(cfg.params["target_frame_name"])

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        frame_cfg: SceneEntityCfg,
        target_frame_name: str,
        axes: str | None = None,
    ) -> torch.Tensor:
        distance = torch.norm(
            self._frame.data.target_pos_source[:, self._target_frame_idx, self._axes],
            dim=-1,
        )

        return 1 - torch.tanh(distance / std)


class orientation_rel_reward_tanh(ManagerTermBase):
    """Reward based on the relative orientation of target frames with respect to source frame in frame_cfg using tanh
    kernel. If multiple target frames are provided, the reward is the mean of the rewards for each target frame."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        # Get the frame_cfg
        self._frame = env.scene[cfg.params["frame_cfg"].name]

        # Get the indices of the target frames in the frame_cfg
        self._target_frame_idx = self._frame._target_frame_names.index(cfg.params["target_frame_name"])

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        frame_cfg: SceneEntityCfg,
        target_frame_name: str,
    ) -> torch.Tensor:
        quat = self._frame.data.target_quat_source[:, self._target_frame_idx]
        # Penalize the norm of the axis-angle representation of the relative orientation
        return 1 - torch.tanh(torch.norm(math_utils.axis_angle_from_quat(quat), dim=-1) / std)


class heading_rel_reward_tanh(orientation_rel_reward_tanh):
    """Reward based on the relative heading of target frames with respect to source frame in frame_cfg using tanh
    kernel. If multiple target frames are provided, the reward is the mean of the rewards for each target frame."""

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        frame_cfg: SceneEntityCfg,
        target_frame_name: str,
    ) -> torch.Tensor:
        yaw_quat = math_utils.yaw_quat(self._frame.data.target_quat_source[:, self._target_frame_idx])
        # Penalize the norm of the axis-angle representation of the relative orientation
        return 1 - torch.tanh(torch.norm(math_utils.axis_angle_from_quat(yaw_quat), dim=-1) / std)
