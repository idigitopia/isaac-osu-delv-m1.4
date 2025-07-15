# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing task implementations for various robotic environments."""

import traceback

import omni.log
from isaaclab_tasks.utils import import_packages

# The blacklist is used to prevent importing configs from sub-packages
_BLACKLIST_PKGS = ["utils", ".mdp", "scripts", "data", "robot_configs"]
# Import all configs in this package
try:
    import_packages(__name__, _BLACKLIST_PKGS)
except ImportError as e:  # noqa: F841
    omni.log.error(f"Failed to load some packages from {__name__} please debug this locally.")
    omni.log.error(traceback.format_exc())
