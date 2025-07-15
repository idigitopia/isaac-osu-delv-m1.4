# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to enable different events.

Events include anything related to altering the simulation state. This includes changing the physics
materials, applying external forces, and resetting the state of the asset.

The functions can be passed to the :class:`isaaclab.managers.EventTermCfg` object to enable
the event introduced by the function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation, RigidObject, RigidObjectCollection
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import GREEN_ARROW_X_MARKER_CFG

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


class visualize_asset_pose(ManagerTermBase):
    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self._debug_vis = cfg.params.get("debug_vis", False)

        if self._debug_vis:
            goal_pose_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
                prim_path="/Visuals/Command/pose_goal"
            )
            goal_pose_visualizer_cfg.markers["arrow"].scale = cfg.params["marker_scale"]
            goal_pose_visualizer_cfg.markers["arrow"].visual_material.diffuse_color = cfg.params["marker_color"]
            self.goal_pose_visualizer = VisualizationMarkers(goal_pose_visualizer_cfg)
            self.goal_pose_visualizer.set_visibility(True)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        marker_scale: tuple[float, float, float] = (0.2, 0.2, 0.8),
        marker_color: tuple[float, float, float] = (0.0, 1.0, 0.0),
        debug_vis: bool = False,
    ):
        if not self._debug_vis:
            return

        asset: RigidObject | Articulation = env.scene[asset_cfg.name]

        positions = asset.data.root_pos_w.clone()
        positions[:, 2] += 0.2

        self.goal_pose_visualizer.visualize(
            translations=positions,
            orientations=asset.data.root_quat_w,
        )


def reset_root_state_uniform_restricted(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    asset_cfg_restricted: SceneEntityCfg = None,
    min_distance: float = 0.5,
    max_attempts: int = 100,
):
    """Reset the asset root state to a random position and velocity uniformly within the given ranges.

    Ensures that the sampled position is not too close to asset_other's root position.
    """
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    root_states = asset.data.default_root_state[env_ids].clone()

    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)

    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)
    sampled_positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]

    if asset_cfg_restricted:
        other_asset = env.scene[asset_cfg_restricted.name]
        restricted_asset_pos = other_asset.data.root_pos_w[env_ids, :3]

        distances = torch.norm(sampled_positions - restricted_asset_pos, dim=1)
        invalid_positions = distances < min_distance

        attempts = 0
        while invalid_positions.any() and attempts < max_attempts:
            new_samples = math_utils.sample_uniform(
                ranges[:, 0], ranges[:, 1], (invalid_positions.sum(), 6), device=asset.device
            )
            sampled_positions[invalid_positions] = (
                root_states[invalid_positions, 0:3]
                + env.scene.env_origins[env_ids[invalid_positions]]
                + new_samples[:, 0:3]
            )
            distances = torch.norm(sampled_positions - restricted_asset_pos, dim=1)
            invalid_positions = distances < min_distance
            attempts += 1
        if attempts == max_attempts:
            print("[WARNING] Failed to sample valid positions for all environments")

    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)

    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)
    velocities = root_states[:, 7:13] + rand_samples

    asset.write_root_pose_to_sim(torch.cat([sampled_positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def reset_batch_root_state_uniform_restricted(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    # TODO: Figure out way to make this a list and prevent error from hydra
    asset1_cfg_restricted: SceneEntityCfg | None = None,
    asset2_cfg_restricted: SceneEntityCfg | None = None,
    min_distance: float = 0.5,
    max_attempts: int = 100,
):
    asset_collection = env.scene[asset_cfg.name]
    num_objects = asset_collection.num_objects
    root_states = asset_collection.data.default_object_state[env_ids].clone()

    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset_collection.device)

    rand_samples = math_utils.sample_uniform(
        ranges[:, 0], ranges[:, 1], (len(env_ids), num_objects, 6), device=asset_collection.device
    )
    sampled_positions = root_states[:, :, 0:3] + env.scene.env_origins[env_ids].unsqueeze(1) + rand_samples[:, :, 0:3]

    asset_cfgs_restricted = [asset1_cfg_restricted, asset2_cfg_restricted]
    for asset_cfg_restricted in asset_cfgs_restricted:
        if asset_cfg_restricted is None:
            continue
        other_asset = env.scene[asset_cfg_restricted.name]
        restricted_asset_pos = other_asset.data.root_pos_w[env_ids, :3]

        distances = torch.norm(sampled_positions - restricted_asset_pos.unsqueeze(1), dim=2)
        invalid_positions = distances < min_distance

        attempts = 0
        while invalid_positions.any() and attempts < max_attempts:
            new_samples = math_utils.sample_uniform(
                ranges[:, 0], ranges[:, 1], (invalid_positions.sum(), 6), device=asset_collection.device
            )
            sampled_positions[invalid_positions] = (
                root_states[invalid_positions][:, 0:3]
                + env.scene.env_origins[env_ids.unsqueeze(1).expand(-1, num_objects)][invalid_positions]
                + new_samples[:, 0:3]
            )
            distances = torch.norm(sampled_positions - restricted_asset_pos.unsqueeze(1), dim=2)
            invalid_positions = distances < min_distance
            attempts += 1
        if attempts == max_attempts:
            print("[WARNING] Failed to sample valid positions for all environments")

    orientations_delta = math_utils.quat_from_euler_xyz(
        rand_samples[:, :, 3], rand_samples[:, :, 4], rand_samples[:, :, 5]
    )
    orientations = math_utils.quat_mul(root_states[:, :, 3:7], orientations_delta)

    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset_collection.device)
    rand_samples = math_utils.sample_uniform(
        ranges[:, 0], ranges[:, 1], (len(env_ids), num_objects, 6), device=asset_collection.device
    )
    velocities = root_states[:, :, 7:13] + rand_samples

    asset_collection.write_object_pose_to_sim(torch.cat([sampled_positions, orientations], dim=-1), env_ids=env_ids)
    asset_collection.write_object_velocity_to_sim(velocities, env_ids=env_ids)


def move_asset_by_command(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
):
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    current_positions = asset.data.root_pos_w[:, :3].clone()
    current_orientations = asset.data.root_quat_w[:, :4].clone()

    se3_command_delta = env.command_manager.get_term(command_name)._s3_command_delta

    se3_command_pos_delta = torch.tensor(
        [se3_command_delta[0], se3_command_delta[1], 0.0], device=asset.device
    ).unsqueeze(0)

    se3_command_quat_delta = math_utils.quat_from_euler_xyz(
        torch.tensor([0.0], device=asset.device),
        torch.tensor([0.0], device=asset.device),
        torch.tensor([se3_command_delta[2]], device=asset.device),
    ).expand(len(env_ids), -1)

    asset.write_root_pose_to_sim(
        torch.cat(
            [
                current_positions + se3_command_pos_delta,
                math_utils.quat_mul(current_orientations, se3_command_quat_delta),
            ],
            dim=-1,
        ),
        env_ids=env_ids,
    )


def invoke_obs_term_func(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    group_name: str = "policy",
    term_name: str = "base_lin_vel",
    func_name: str = "set_mask",
    func_kwargs: dict | None = None,
):
    obs_group = env.observation_manager._group_obs_term_cfgs[group_name]
    obs_term_cfg = obs_group[env.observation_manager._group_obs_term_names[group_name].index(term_name)]

    func = getattr(obs_term_cfg.func, func_name)
    if func_kwargs is not None:
        func(env, env_ids, **func_kwargs)
    else:
        func(env, env_ids)


def push_by_setting_velocity_probab(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    probability: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Push the asset by setting the root velocity to a random value within the given ranges.

    This creates an effect similar to pushing the asset with a random impulse that changes the asset's velocity.
    It samples the root velocity from the given ranges and sets the velocity into the physics simulation.

    The function takes a dictionary of velocity ranges for each axis and rotation. The keys of the dictionary
    are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form ``(min, max)``.
    If the dictionary does not contain a key, the velocity is set to zero for that axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # velocities
    vel_w = asset.data.root_vel_w[env_ids]
    # sample random velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    velocities = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], vel_w.shape, device=asset.device)

    # sample the probability
    apply_vel = torch.rand(len(env_ids), device=asset.device) < probability
    velocities[~apply_vel] = 0.0
    vel_w += velocities
    # set the velocities into the physics simulation
    asset.write_root_velocity_to_sim(vel_w, env_ids=env_ids)


def apply_external_force_torque_probab(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    force_range: tuple[float, float],
    torque_range: tuple[float, float],
    probability: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Randomize the external forces and torques applied to the bodies.

    This function creates a set of random forces and torques sampled from the given ranges. The number of forces
    and torques is equal to the number of bodies times the number of environments. The forces and torques are
    applied to the bodies by calling ``asset.set_external_force_and_torque``. The forces and torques are only
    applied when ``asset.write_data_to_sim()`` is called in the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    # resolve number of bodies
    num_bodies = len(asset_cfg.body_ids) if isinstance(asset_cfg.body_ids, list) else asset.num_bodies

    # sample random forces and torques
    size = (len(env_ids), num_bodies, 3)
    forces = math_utils.sample_uniform(*force_range, size, asset.device)
    torques = math_utils.sample_uniform(*torque_range, size, asset.device)

    # sample the probability
    apply_forces = torch.rand(len(env_ids), device=asset.device) < probability
    apply_torques = torch.rand(len(env_ids), device=asset.device) < probability
    forces[~apply_forces] = 0.0
    torques[~apply_torques] = 0.0
    # set the forces and torques into the buffers
    # note: these are only applied when you call: `asset.write_data_to_sim()`
    asset.set_external_force_and_torque(forces, torques, env_ids=env_ids, body_ids=asset_cfg.body_ids)


def reset_object_pose_uniform(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("objects"),
):
    """Reset the object pose to a random position and orientation uniformly within the given ranges.

    The function takes a dictionary of pose ranges for each axis and rotation in 2D plane. The keys of the dictionary
    are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form ``(min, max)``.
    If the dictionary does not contain a key, the pose is set to zero for that axis.
    """

    asset: RigidObject | RigidObjectCollection = env.scene[asset_cfg.name]
    # get the default object state
    object_states = asset.data.default_object_state[env_ids].clone()

    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    # if asset is rigid object then sample one pose, if it is rigid object collection then sample multiple poses
    if isinstance(asset, RigidObject):
        rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)
        positions = object_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
        orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        orientations = math_utils.quat_mul(object_states[:, 3:7], orientations_delta)

        asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    else:
        rand_samples = math_utils.sample_uniform(
            ranges[:, 0], ranges[:, 1], (len(env_ids), asset.num_objects, 6), device=asset.device
        )
        positions = object_states[:, :, 0:3] + env.scene.env_origins[env_ids].unsqueeze(1) + rand_samples[:, :, 0:3]
        orientations_delta = math_utils.quat_from_euler_xyz(
            rand_samples[:, :, 3], rand_samples[:, :, 4], rand_samples[:, :, 5]
        )
        orientations = math_utils.quat_mul(object_states[:, :, 3:7], orientations_delta)

        asset.write_object_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
