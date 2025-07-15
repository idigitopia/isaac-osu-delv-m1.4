from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

try:
    from isaaclab.devices import Se3Keyboard
except AttributeError:
    print(
        "[ERROR] Se3Keyboard cannot be imported. If you are running this script in a headless mode, this is expected."
    )


# TODO: Handle any number of restricted assets and better way to check if the position is valid
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


class move_asset_by_gui_keyboard(ManagerTermBase):
    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        assert cfg.mode == "interval", "move_asset_by_gui_keyboard must be used in interval mode"
        assert cfg.interval_range_s == (0.0, 0.0), "interval_range_s must be (0.0, 0.0)"

        self._asset = env.scene[cfg.params["asset_cfg"].name]
        self._offset = torch.tensor(cfg.params["offset"], device=env.device).unsqueeze(0)

        self._teleop_interface = Se3Keyboard(
            pos_sensitivity=cfg.params["pos_sensitivity"], rot_sensitivity=cfg.params["rot_sensitivity"]
        )
        self._teleop_interface.add_callback("R", self._env.reset)

        print("Teleop interface: ", self._teleop_interface)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        offset: tuple[float, float, float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        pos_sensitivity: float = 0.05,
        rot_sensitivity: float = 0.05,
    ):
        command, _ = self._teleop_interface.advance()
        se3_command_delta = (
            torch.from_numpy(command).to(env.device, dtype=torch.float32).unsqueeze(0).expand(len(env_ids), -1)
        )

        # b: box frame
        # o: offset frame
        # b_prime: box frame after apply delta command to its offset
        # o_prime: offset frame after apply delta command

        # T_o_o_prime
        delta_pos = se3_command_delta[:, 0:3]
        delta_quat = math_utils.quat_from_euler_xyz(
            roll=se3_command_delta[:, 3], pitch=se3_command_delta[:, 4], yaw=se3_command_delta[:, 5]
        )

        # T_w_b
        pos_wb = self._asset.data.root_pos_w[env_ids]
        quat_wb = self._asset.data.root_quat_w[env_ids]

        # T_b_o
        pos_bo = self._offset[:, :3].expand(len(env_ids), -1)
        quat_bo = self._offset[:, 3:].expand(len(env_ids), -1)

        # T_w_o = T_w_b * T_b_o
        pos_wo, quat_wo = math_utils.combine_frame_transforms(pos_wb, quat_wb, pos_bo, quat_bo)

        # T_w_o_prime = T_w_o * T_o_o_prime
        pos_wo_prime, quat_wo_prime = math_utils.combine_frame_transforms(pos_wo, quat_wo, delta_pos, delta_quat)

        # T_o_prime_b_prime = inv(T_b_o)
        pos_o_prime_b_prime, quat_o_prime_b_prime = math_utils.subtract_frame_transforms(pos_bo, quat_bo)

        # T_w_b_prime = T_w_o_prime * T_o_prime_b_prime
        pos_wb_prime, quat_wb_prime = math_utils.combine_frame_transforms(
            pos_wo_prime, quat_wo_prime, pos_o_prime_b_prime, quat_o_prime_b_prime
        )

        # Write updated pose to sim
        self._asset.write_root_pose_to_sim(
            torch.cat((pos_wb_prime, quat_wb_prime), dim=-1),
            env_ids=env_ids,
        )
