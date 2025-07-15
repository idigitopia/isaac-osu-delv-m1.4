# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab.utils.noise import NoiseCfg

from . import noise_model


@configclass
class SaltPepperNoiseCfg(NoiseCfg):
    """Configuration for salt-and-pepper noise."""

    func = noise_model.salt_and_pepper_noise

    probability: float = 0.01

    replacement_value_range: tuple[float, float] = (0.0, 1.0)
