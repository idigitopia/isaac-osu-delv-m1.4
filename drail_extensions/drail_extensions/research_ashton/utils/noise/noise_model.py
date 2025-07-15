# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from . import noise_cfg

##
# Noise as functions.
##


def salt_and_pepper_noise(data: torch.Tensor, cfg: noise_cfg.SaltPepperNoiseCfg) -> torch.Tensor:
    """Applies salt-and-pepper noise to simulate missing depth points."""
    mask = torch.rand_like(data) < cfg.probability
    # Use randint to randomly select between 0 and 1, then convert to float
    data[mask] = (
        torch.rand(data[mask].shape, device=data.device)
        * (cfg.replacement_value_range[1] - cfg.replacement_value_range[0])
        + cfg.replacement_value_range[0]
    )
    return data
