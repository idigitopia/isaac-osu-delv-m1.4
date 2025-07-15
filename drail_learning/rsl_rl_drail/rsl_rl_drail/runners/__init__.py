# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implementation of runners for environment-agent interaction."""

from .on_policy_runner import OnPolicyRunner  # isort: skip

# TODO: Have to import AMPOnPolicyRunner *AFTER* OnPolicyRunner because it itself imports and uses
# OnPolicyRunner. So need to fully initialize on_policy_runner module first. This kind of circular
# import is hacky, can maybe look to fix later
from .amp_on_policy_runner import AMPOnPolicyRunner

__all__ = ["OnPolicyRunner", "AMPOnPolicyRunner"]
