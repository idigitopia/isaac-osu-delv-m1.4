# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create observation terms.

The functions can be passed to the :class:`isaaclab.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg

import drail_extensions.core.utils.math_utils as core_math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def body_height_w(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Body height in env frame."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - env.scene.env_origins[:, 2].unsqueeze(1)


class pos_rel(ManagerTermBase):
    """Returns the position of the target frames relative to the environment frame. For multiple frames,
    the relative positions are concatenated along the last dimension."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._frame = env.scene[cfg.params["frame_cfg"].name]
        self._axes = torch.tensor(["xyz".index(ax) for ax in cfg.params.get("axes", "xyz")], device=self.device).long()
        self._target_frame_indices = torch.tensor(
            [self._frame._target_frame_names.index(name) for name in cfg.params["target_frame_names"]],
            device=self.device,
        ).long()

    def __call__(
        self, env: ManagerBasedEnv, frame_cfg: SceneEntityCfg, target_frame_names: list[str], axes: str | None = None
    ) -> torch.Tensor:
        return (
            self._frame.data.target_pos_source.index_select(dim=1, index=self._target_frame_indices)
            .index_select(dim=-1, index=self._axes)
            .flatten(1)
        )


class quat_rel(ManagerTermBase):
    """Returns the quaternion of the target frames relative to the environment frame. For multiple frames,
    the relative quaternions are concatenated along the last dimension."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._frame = env.scene[cfg.params["frame_cfg"].name]

        # Get the indices of the target frames in the frame_cfg
        self._target_frame_indices = [
            self._frame._target_frame_names.index(name) for name in cfg.params["target_frame_names"]
        ]

    def __call__(
        self, env: ManagerBasedEnv, frame_cfg: SceneEntityCfg, target_frame_names: list[str], euler: bool = False
    ) -> torch.Tensor:
        quat = self._frame.data.target_quat_source[:, self._target_frame_indices]
        if euler:
            return core_math_utils.euler_xyz_from_quat(quat).flatten(1)
        return quat.flatten(1)


class heading_rel(quat_rel):
    """Returns the heading of the target frames relative to the environment frame."""

    def __call__(
        self, env: ManagerBasedEnv, frame_cfg: SceneEntityCfg, target_frame_names: list[str], euler: bool = False
    ) -> torch.Tensor:
        yaw_quat = math_utils.yaw_quat(self._frame.data.target_quat_source[:, self._target_frame_indices])
        if euler:
            return core_math_utils.euler_xyz_from_quat(yaw_quat)[..., 2].flatten(1)
        return yaw_quat.flatten(1)
