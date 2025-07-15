# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os

from rsl_rl.utils.wandb_utils import WandbSummaryWriter as BaseWandbSummaryWriter
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
except ModuleNotFoundError:
    raise ModuleNotFoundError("Wandb is required to log to Weights and Biases.")


class WandbSummaryWriter(BaseWandbSummaryWriter):
    """Summary writer for Weights and Biases."""

    def __init__(self, log_dir: str, flush_secs: int, cfg):
        SummaryWriter.__init__(self, log_dir, flush_secs)

        try:
            project = cfg["wandb_project"]
        except KeyError:
            raise KeyError("Please specify wandb_project in the runner config, e.g. legged_gym.")

        try:
            entity = os.environ["WANDB_USERNAME"]
        except KeyError:
            raise KeyError(
                "Wandb username not found. Please run or add to ~/.bashrc: export WANDB_USERNAME=YOUR_USERNAME"
            )
        group = cfg["experiment_name"] if "experiment_name" in cfg else None
        name = cfg["run_name"] if "run_name" in cfg and cfg["run_name"] != "" else None

        self.run = wandb.init(project=project, entity=entity, group=group)

        # Change generated name to project-number format
        if wandb.run.name is not None:  # Support wandb offline mode
            if name is None:
                run_name_prefix = project if name is None else name
                run_name_suffix = wandb.run.name.split("-")[-1]
                wandb.run.name = run_name_prefix + "_" + run_name_suffix
            else:
                wandb.run.name = name

        self.name_map = {
            "Train/mean_reward/time": "Train/mean_reward_time",
            "Train/mean_episode_length/time": "Train/mean_episode_length_time",
        }

        run_name = os.path.split(log_dir)[-1]

        wandb.log({"log_dir": run_name})

    def _map_path(self, path):
        if path in self.name_map:
            return self.name_map[path]
        return path
