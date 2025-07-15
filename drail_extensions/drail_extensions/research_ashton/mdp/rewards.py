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
from isaaclab.assets import RigidObject, Articulation
from isaaclab.sensors import ContactSensor
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

import isaaclab.utils.math as math_utils
import numpy as np
from isaaclab.managers import ManagerTermBase
from isaaclab.managers.manager_term_cfg import RewardTermCfg


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

    return (curr_height < min_height) | (curr_height > max_height)


def body_height_out_of_range(
    env: ManagerBasedRLEnv,
    min_height: float = -np.inf,
    max_height: float = np.inf,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = None,
) -> torch.Tensor:
    """Penalize body height from its target using exponential kernel."""
    asset: RigidObject = env.scene[asset_cfg.name]

    curr_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]

    return torch.any(torch.logical_or(curr_height < min_height, curr_height > max_height), dim=1)


def track_base_height_exp(
    env: ManagerBasedRLEnv,
    target_height: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = None,
    command_name: str = None,
) -> torch.Tensor:
    """Reward tracking nominal target base height or commanded using exponential kernel."""

    asset: RigidObject = env.scene[asset_cfg.name]

    # get commanded height
    if command_name is not None:
        target_height = env.command_manager.get_command(command_name)[:, 4]

    # Adjust the target height based on the sensor readings for rough terrain
    if sensor_cfg is not None:
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        target_height += torch.mean(contact_sensor.data.ray_hits_w[..., 2], dim=1)

    curr_height = asset.data.root_pos_w[:, 2]
    height_error = torch.abs(curr_height - target_height)

    return torch.exp(-height_error / std**2)


def track_body_tilt_error(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Quaternion error between commanded and current body tilt."""

    asset: RigidObject = env.scene[asset_cfg.name]

    # get commanded tilt
    command = env.command_manager.get_command(command_name)[:, 5:]

    body_quat = asset.data.body_quat_w[:, asset_cfg.body_ids, :].squeeze(1)
    # get euler angles from the quaternion
    roll, pitch, _ = math_utils.euler_xyz_from_quat(body_quat)
    # wrap to pi
    roll = math_utils.wrap_to_pi(roll)
    pitch = math_utils.wrap_to_pi(pitch)

    # error
    roll_err = torch.abs(command[:, 0] - roll)
    pitch_err = torch.abs(command[:, 1] - pitch)

    return torch.exp(-(roll_err + pitch_err)/ std**2)


def track_lin_vel_xy_exp_spot(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    std: float, ramp_at_vel: float = 1.0, ramp_rate: float = 0.5
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

    return torch.exp(-lin_vel_error / std**2) * velocity_scaling_multiple


def track_ang_vel_z_exp_spot(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    target = env.command_manager.get_command(command_name)[:, 2]
    # compute the error
    ang_vel_error = torch.linalg.norm((target - asset.data.root_ang_vel_b[:, 2]).unsqueeze(1), dim=1)

    return torch.exp(-ang_vel_error / std**2)


def track_lin_vel_x_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = torch.square(
        torch.abs(env.command_manager.get_command(command_name)[:, 0] - asset.data.root_lin_vel_b[:, 0]),
    )
    return torch.exp(-lin_vel_error / std**2)


def track_lin_vel_y_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = torch.square(
        torch.abs(env.command_manager.get_command(command_name)[:, 1] - asset.data.root_lin_vel_b[:, 1]),
    )
    return torch.exp(-lin_vel_error / std**2)


def base_motion_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, lin_weight: float) -> torch.Tensor:
    """Penalize base vertical and roll/pitch velocity"""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    return lin_weight * torch.square(asset.data.root_lin_vel_b[:, 2]) + \
        (1 - lin_weight) * torch.sum(torch.abs(asset.data.root_ang_vel_b[:, :2]), dim=1)


def base_orientation_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize non-flat base orientation

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.linalg.norm((asset.data.projected_gravity_b[:, :2]), dim=1)


def joint_acceleration_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint accelerations on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.linalg.norm((asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def joint_torques_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint torques on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.linalg.norm((asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)


def joint_velocity_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint velocities on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.linalg.norm((asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def joint_position_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str,
    stand_still_scale: float, velocity_threshold: float
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name)[:, :3], dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)

    reward = torch.linalg.norm(
        (asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]),
        dim=1)

    return torch.where(
        torch.logical_or(cmd > 0.0, body_vel > velocity_threshold),
        reward,
        stand_still_scale * reward
    )


def action_smoothness_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize large instantaneous changes in the network action output"""
    return torch.linalg.norm(
        (env.action_manager.action - env.action_manager.prev_action), dim=1
    )


def foot_clearance_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""

    vel_cmd = torch.norm(env.command_manager.get_command(command_name)[:, :3], dim=1) > 0.0

    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh

    return torch.exp(-torch.sum(reward, dim=1) / std**2) * vel_cmd


def foot_slip_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Penalize foot planar (xy) slip when in contact with the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    foot_planar_velocity = torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)

    reward = is_contact * foot_planar_velocity
    return torch.sum(reward, dim=1)


def feet_air_time_v2(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

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
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :3], dim=1) > 0.0

    return reward


def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, command_name: str) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    cmd = torch.norm(env.command_manager.get_command(command_name)[:, :3], dim=1)
    # Reward only when the command is non-zero
    return (
        torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(torch.clip(last_contact_time, max=0.5), dim=1)
    ) * (cmd > 0.0)


class GaitReward(ManagerTermBase):
    """Gait enforcing reward term for quadrupeds.

    This reward penalizes contact timing differences between selected foot pairs defined in :attr:`synced_feet_pair_names`
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
        command_name: str,
        std: float,
        max_err: float,
        synced_feet_pair_names,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
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
        cmd = torch.norm(env.command_manager.get_command(command_name)[:, :3], dim=1)

        return sync_reward * async_reward * (cmd > 0.0)

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
        return torch.exp(-(se_air + se_contact) / self.std**2)

    def _async_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward anti-synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between opposing contact modes air time of feet 1 to contact time of feet 2
        # and contact time of feet 1 to air time of feet 2) of feet pairs that are not in sync with each other.
        se_act_0 = torch.clip(torch.square(air_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        se_act_1 = torch.clip(torch.square(contact_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_act_0 + se_act_1) / self.std**2)


def feet_contact_quadruped(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg
    ) -> torch.Tensor:
    """Reward all feet in contact when the command is zero"""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # If foot is in contact in current step
    contact = (contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids].norm(dim=-1) >= 0.1).any(dim=1)

    # All feet in contact
    reward = contact.sum(dim=-1) == 4

    zero_command = torch.norm(env.command_manager.get_command(command_name)[:, :3], dim=1) == 0

    # If the command is zero, reward feet contact
    return reward * zero_command


def feet_contact_quadruped_spot(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    velocity_threshold: float = 0.25
    ) -> torch.Tensor:
    """Reward more feet in contact when the command is zero or body velocity is below a threshold."""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # If foot is in contact in current step
    contact = (contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids].norm(dim=-1) >= 0.1).any(dim=1)

    # Soft reward just rewarding more feet in contact
    reward = contact.sum(dim=-1) / contact.shape[-1]

    cmd = torch.norm(env.command_manager.get_command(command_name)[:, :3], dim=1)
    body_vel = torch.linalg.norm(env.scene[asset_cfg.name].data.root_lin_vel_b[:, :2], dim=1)

    reward = torch.where(
        torch.logical_or(cmd > 0.0, body_vel > velocity_threshold),
        0.0,
        reward
    )

    return reward


def stance_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    std: float,
    x_dist: float = 0.38,
    y_dist: float = 0.27,
    diag_dist: float = 0.47,
    ) -> torch.Tensor:
    """Penalize the stance of the asset."""

    asset: RigidObject = env.scene[asset_cfg.name]

    feet_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids][..., :2]

    # 0 - Front Left, 1 - Front Right, 2 - Rear Left, 3 - Rear Right
    # Distance between front left and front right feet
    fl_fr = torch.abs(torch.norm(feet_pos_w[:, 0] - feet_pos_w[:, 1], dim=-1) - y_dist)
    # Distance between rear left and rear right feet
    rl_rr = torch.abs(torch.norm(feet_pos_w[:, 2] - feet_pos_w[:, 3], dim=-1) - y_dist)
    # Distance between front left and rear left feet
    fl_rl = torch.abs(torch.norm(feet_pos_w[:, 0] - feet_pos_w[:, 2], dim=-1) - x_dist)
    # Distance between front right and rear right feet
    fr_rr = torch.abs(torch.norm(feet_pos_w[:, 1] - feet_pos_w[:, 3], dim=-1) - x_dist)
    # # Distance between front left and rear right feet
    # fl_rr = torch.abs(torch.norm(feet_pos_w[:, 0] - feet_pos_w[:, 3], dim=-1) - diag_dist)
    # # Distance between front right and rear left feet
    # fr_rl = torch.abs(torch.norm(feet_pos_w[:, 1] - feet_pos_w[:, 2], dim=-1) - diag_dist)

    zero_command = torch.norm(env.command_manager.get_command(command_name)[:, :3], dim=1) < 0.05

    stance_error = fl_fr + rl_rr + fl_rl + fr_rr #+ fl_rr + fr_rl

    return torch.exp(-stance_error / std**2) * zero_command


def joint_deviation_exp(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    std: float,
    stand_std: float,
    ) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one with a exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    angle = torch.linalg.norm(
        (asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]),
        dim=1
    )

    # get zero command
    zero_command = torch.norm(env.command_manager.get_command(command_name)[:, :3], dim=1) < 0.05

    # scale the error based on zero command environment
    angle = torch.where(
        zero_command,
        angle / stand_std**2,
        angle / std**2
    )

    return torch.exp(-angle)


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


def delta_pos_command_exp(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg, std: float) -> torch.Tensor:
    """Penalize tracking of the position error using exponential kernel of L2-norm.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame). The position error is computed as the L2-norm
    of the difference between the desired and current positions.
    """
    # extract the asset (to enable type hinting)
    # asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :2]
    # des_pos_w, _ = math_utils.combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
    # curr_pos_w = asset.data.root_state_w[:, :3]
    error = torch.norm(des_pos_b, dim=1)

    return torch.exp(-error / std**2)


def delta_orient_command_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Penalize tracking orientation error."""
    command = env.command_manager.get_command(command_name)
    heading_b = command[:, 3]

    return torch.exp(-heading_b.abs() / std**2)


def position_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame). The position error is computed as the L2-norm
    of the difference between the desired and current positions.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = math_utils.combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
    return torch.norm(curr_pos_w - des_pos_w, dim=1)


def position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of the position using the tanh kernel.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame) and maps it with a tanh kernel.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = math_utils.combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return 1 - torch.tanh(distance / std)


def orientation_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking orientation error using shortest path.

    The function computes the orientation error between the desired orientation (from the command) and the
    current orientation of the asset's body (in world frame). The orientation error is computed as the shortest
    path between the desired and current orientations.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current orientations
    des_quat_b = command[:, 3:7]
    des_quat_w = math_utils.quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)
    curr_quat_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]  # type: ignore
    return math_utils.quat_error_magnitude(curr_quat_w, des_quat_w)


def heading_command_error_abs(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Penalize tracking orientation error."""
    command = env.command_manager.get_command(command_name)
    heading_b = command[:, 3]
    return heading_b.abs()


def position_rel_reward_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    offset_d: float = 0.0,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    offset_z: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    asset_cfg_other: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward position tracking with tanh kernel, but decreases if too close to optimal distance.
    which is based on the relative pose of asset_cfg_other with respect to asset_cfg.
    """

    asset: RigidObject = env.scene[asset_cfg.name]
    asset_other: RigidObject = env.scene[asset_cfg_other.name]

    target_vec_w = asset_other.data.root_pos_w[:, :3] - asset.data.root_pos_w[:, :3]
    target_vec_b = math_utils.quat_rotate_inverse(math_utils.yaw_quat(asset.data.root_quat_w), target_vec_w)

    des_pos_b = target_vec_b[:, :3]

    des_pos_b[:, 0] += offset_x
    des_pos_b[:, 1] += offset_y
    des_pos_b[:, 2] += offset_z

    distance = torch.norm(des_pos_b, dim=1)

    return 1 - torch.tanh((distance - offset_d).abs() / std)


def heading_rel_error_abs(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    offset_heading: float = 0.0,
    asset_cfg_other: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Penalize tracking orientation error based on the relative pose of asset_cfg_other with respect to asset_cfg."""
    asset: RigidObject = env.scene[asset_cfg.name]
    asset_other: RigidObject = env.scene[asset_cfg_other.name]

    heading_b = math_utils.wrap_to_pi(asset_other.data.heading_w - asset.data.heading_w + offset_heading)

    return heading_b.abs()
