# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

import math

import isaaclab.utils.math as math_utils

"""
MDP terminations.
"""


def illegal_contact_with_filter(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Terminate when the contact force with filtered bodies exceeds the force threshold."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.force_matrix_w
    # check if any contact force exceeds the threshold with any of the filtered bodies
    return torch.any(torch.norm(net_contact_forces[:, sensor_cfg.body_ids], dim=-1) > threshold, dim=(1, 2))


def rpy_limit(
    env: ManagerBasedRLEnv,
    roll_limit: float = (-math.inf, math.inf),
    pitch_limit: float = (-math.inf, math.inf),
    yaw_limit: float = (-math.inf, math.inf),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the RPY limit is exceeded."""
    asset: Articulation = env.scene[asset_cfg.name]

    roll, pitch, yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)

    roll = math_utils.wrap_to_pi(roll)
    pitch = math_utils.wrap_to_pi(pitch)
    yaw = math_utils.wrap_to_pi(yaw)

    return (
        (roll < roll_limit[0])
        | (roll > roll_limit[1])
        | (pitch < pitch_limit[0])
        | (pitch > pitch_limit[1])
        | (yaw < yaw_limit[0])
        | (yaw > yaw_limit[1])
    )
