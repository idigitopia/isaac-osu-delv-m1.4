# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination manager for computing done signals for a given world."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.managers.manager_base import ManagerBase, ManagerTermBase
from isaaclab.managers.manager_term_cfg import TerminationTermCfg
from prettytable import PrettyTable

import drail_extensions.core.utils.string as string_utils
from drail_extensions.core.managers import TerminationGroupCfg

if TYPE_CHECKING:
    from drail_extensions.core.envs import ManagerBasedMARLEnv


class TerminationManager(ManagerBase):
    """Manager for computing done signals for a given world.

    The termination manager computes the termination signal (also called dones) as a combination
    of termination terms. Each termination term is a function which takes the environment as an
    argument and returns a boolean tensor of shape (num_envs,). The termination manager
    computes the termination signal as the union (logical or) of all the termination terms.

    Following the `Gymnasium API <https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/>`_,
    the termination signal is computed as the logical OR of the following signals:

    * **Time-out**: This signal is set to true if the environment has ended after an externally defined condition
      (that is outside the scope of a MDP). For example, the environment may be terminated if the episode has
      timed out (i.e. reached max episode length).
    * **Terminated**: This signal is set to true if the environment has reached a terminal state defined by the
      environment. This state may correspond to task success, task failure, robot falling, etc.

    These signals can be individually accessed using the :attr:`time_outs` and :attr:`terminated` properties.

    The termination terms are parsed from a config class containing the manager's settings and each term's
    parameters. Each termination term should instantiate the :class:`TerminationTermCfg` class. The term's
    configuration :attr:`TerminationTermCfg.time_out` decides whether the term is a timeout or a termination term.
    """

    _env: ManagerBasedMARLEnv
    """The environment instance."""

    def __init__(self, cfg: object, env: ManagerBasedMARLEnv):
        """Initializes the termination manager.

        Args:
            cfg: The configuration object or dictionary (``dict[str, TerminationTermCfg]``).
            env: An environment object.
        """
        # create buffers to parse and store terms
        # self._term_names: list[str] = list()
        # self._term_cfgs: list[TerminationTermCfg] = list()
        # self._class_term_cfgs: list[TerminationTermCfg] = list()
        self._group_term_names: dict[str, list[str]] = dict()
        self._group_term_cfgs: dict[str, list[TerminationTermCfg]] = dict()
        self._group_class_term_cfgs: dict[str, list[TerminationTermCfg]] = dict()

        # call the base class constructor (this will parse the terms config)
        super().__init__(cfg, env)
        self._term_dones = dict()
        self._truncated_buf = dict()
        self._terminated_buf = dict()
        self._dones = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        # prepare extra info to store individual termination term information
        for group_name, group_term_names in self._group_term_names.items():
            self._term_dones[group_name] = dict()
            for term_name in group_term_names:
                self._term_dones[group_name][term_name] = torch.zeros(
                    self.num_envs, device=self.device, dtype=torch.bool
                )
            # create buffer for managing termination per environment
            self._truncated_buf[group_name] = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
            self._terminated_buf[group_name] = torch.zeros_like(self._truncated_buf[group_name])

    def __str__(self) -> str:
        """Returns: A string representation for termination manager."""
        msg = f"<TerminationManager> contains {len(self._group_term_names)} active groups.\n"

        for group_name in self._group_term_names.keys():
            # create table for term information
            table = PrettyTable()
            table.title = f"Active Termination Terms in group: {group_name}"
            table.field_names = ["Index", "Name", "Time Out"]
            # set alignment of table columns
            table.align["Name"] = "l"
            # add info on each term
            for index, (name, term_cfg) in enumerate(
                zip(self._group_term_names[group_name], self._group_term_cfgs[group_name])
            ):
                table.add_row([index, name, term_cfg.time_out])
            # convert table to string
            msg += table.get_string()
            msg += "\n"

        msg += string_utils.get_common_terms_repr(self._group_term_names, self._group_term_cfgs)
        return msg

    """
    Properties.
    """

    @property
    def active_terms(self) -> list[str]:
        """Name of active termination terms."""
        return self._group_term_names

    @property
    def dones(self) -> torch.Tensor:
        """The net termination signal across all groups. Shape is (num_envs,)."""
        return self._dones

    @property
    def time_outs(self) -> torch.Tensor:
        """The timeout signal (reaching max episode length). Shape is (num_envs,).

        This signal is set to true if the environment has ended after an externally defined condition
        (that is outside the scope of a MDP). For example, the environment may be terminated if the episode has
        timed out (i.e. reached max episode length).
        """
        return self._truncated_buf

    @property
    def terminated(self) -> torch.Tensor:
        """The terminated signal (reaching a terminal state). Shape is (num_envs,).

        This signal is set to true if the environment has reached a terminal state defined by the environment.
        This state may correspond to task success, task failure, robot falling, etc.
        """
        return self._terminated_buf

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Returns the episodic counts of individual termination terms.

        Args:
            env_ids: The environment ids. Defaults to None, in which case
                all environments are considered.

        Returns:
            Dictionary of episodic sum of individual reward terms.
        """
        # resolve environment ids
        if env_ids is None:
            env_ids = slice(None)
        # add to episode dict
        extras = {}
        for group_name in self._group_term_names.keys():
            # store information
            extras.update(self.reset_group(env_ids, group_name))
        return extras

    def reset_group(
        self, env_ids: Sequence[int] | None = None, group_name: str | None = None
    ) -> dict[str, torch.Tensor]:
        """Returns the episodic counts of individual termination terms.

        Args:
            env_ids: The environment ids. Defaults to None, in which case
                all environments are considered.

        Returns:
            Dictionary of episodic sum of individual reward terms.
        """
        # resolve environment ids
        if env_ids is None:
            env_ids = slice(None)
        # add to episode dict
        extras = {}
        for key in self._term_dones[group_name].keys():
            # store information
            if group_name is None:
                log_key = "Episode_Termination/" + key
            else:
                log_key = f"Episode_Termination/{group_name}/{key}"
            extras[log_key] = torch.count_nonzero(self._term_dones[group_name][key][env_ids]).item()
        # reset all the reward terms
        for term_cfg in self._group_class_term_cfgs[group_name]:
            term_cfg.func.reset(env_ids=env_ids)
        # return logged information
        return extras

    def compute(self) -> torch.Tensor:
        """Computes the termination signal as union of individual terms.

        This function calls each termination term managed by the class and performs a logical OR operation
        to compute the net termination signal.

        Returns:
            The combined termination signal of shape (num_envs,).
        """
        self._dones.fill_(False)
        for group_name in self._group_term_names.keys():
            self._dones |= self.compute_group(group_name)
        return self._dones

    def compute_group(self, group_name: str | None = None) -> torch.Tensor:
        """Computes the termination signal as union of individual terms.

        This function calls each termination term managed by the class and performs a logical OR operation
        to compute the net termination signal.

        Returns:
            The combined termination signal of shape (num_envs,).
        """
        # reset computation
        self._truncated_buf[group_name][:] = False
        self._terminated_buf[group_name][:] = False
        # iterate over all the termination terms
        for name, term_cfg in zip(self._group_term_names[group_name], self._group_term_cfgs[group_name]):
            value = term_cfg.func(self._env, **term_cfg.params)
            # store timeout signal separately
            if term_cfg.time_out:
                self._truncated_buf[group_name] |= value
            else:
                self._terminated_buf[group_name] |= value
            # add to episode dones
            self._term_dones[group_name][name][:] = value
        # return combined termination signal
        return self._truncated_buf[group_name] | self._terminated_buf[group_name]

    def get_term(self, name: str, group_name: str | None = None) -> torch.Tensor:
        """Returns the termination term with the specified name.

        Args:
            name: The name of the termination term.

        Returns:
            The corresponding termination term value. Shape is (num_envs,).
        """
        return self._term_dones[group_name][name]

    def get_active_iterable_terms(self, env_idx: int) -> Sequence[tuple[str, Sequence[float]]]:
        """Returns the active terms as iterable sequence of tuples.

        The first element of the tuple is the name of the term and the second element is the raw value(s) of the term.

        Args:
            env_idx: The specific environment to pull the active terms from.

        Returns:
            The active terms.
        """
        terms = []
        for group_name in self._group_term_names.keys():
            for term_name in self._group_term_names[group_name]:
                terms.append(
                    (
                        group_name + "-" + term_name,
                        [self._term_dones[group_name][term_name][env_idx].float().cpu().item()],
                    )
                )
        return terms

    """
    Operations - Term settings.
    """

    def set_term_cfg(self, term_name: str, cfg: TerminationTermCfg, group_name: str | None = None):
        """Sets the configuration of the specified term into the manager.

        Args:
            term_name: The name of the termination term.
            cfg: The configuration for the termination term.

        Raises:
            ValueError: If the term name is not found.
        """
        if group_name not in self._group_term_names:
            raise ValueError(f"Termination group '{group_name}' not found.")
        if term_name not in self._group_term_names[group_name]:
            raise ValueError(f"Termination term '{term_name}' not found in group '{group_name}'.")
        # set the configuration
        self._group_term_cfgs[group_name][self._group_term_names[group_name].index(term_name)] = cfg

    def get_term_cfg(self, term_name: str, group_name: str | None = None) -> TerminationTermCfg:
        """Gets the configuration for the specified term.

        Args:
            term_name: The name of the termination term.

        Returns:
            The configuration of the termination term.

        Raises:
            ValueError: If the term name is not found.
        """
        if group_name not in self._group_term_names:
            raise ValueError(f"Termination group '{group_name}' not found.")
        if term_name not in self._group_term_names[group_name]:
            raise ValueError(f"Termination term '{term_name}' not found in group '{group_name}'.")
        # return the configuration
        return self._group_term_cfgs[group_name][self._group_term_names[group_name].index(term_name)]

    """
    Helper functions.
    """

    def _prepare_terms(self):
        # check if config is dict already
        if isinstance(self.cfg, dict):
            cfg_items = self.cfg.items()
        else:
            cfg_items = self.cfg.__dict__.items()
        num_groups = sum(isinstance(cfg, TerminationGroupCfg) for _, cfg in cfg_items)
        assert num_groups == 0 or num_groups == len(
            cfg_items
        ), "Either all reward term should have a group or none of them should have a group"

        if num_groups == 0:
            # create a dummy group and add all the terms to it
            dummy_group = TerminationGroupCfg()
            for name, cfg in cfg_items:
                setattr(dummy_group, name, cfg)
            group_cfg_items = {None: dummy_group}
        else:
            group_cfg_items = cfg_items

        for group_name, group_cfg in group_cfg_items:
            # check for non config
            if group_cfg is None:
                continue
            # check for valid config type
            if not isinstance(group_cfg, TerminationGroupCfg):
                raise TypeError(
                    f"Configuration for the group '{group_name}' is not of type TerminationGroupCfg."
                    f" Received: '{type(group_cfg)}'."
                )
            self._group_term_names[group_name] = list()
            self._group_term_cfgs[group_name] = list()
            self._group_class_term_cfgs[group_name] = list()

            # check if config is dict already
            if isinstance(group_cfg, dict):
                cfg_items = group_cfg.items()
            else:
                cfg_items = group_cfg.__dict__.items()

            # iterate over all the terms
            for term_name, term_cfg in cfg_items:
                # check for non config
                if term_cfg is None:
                    continue
                # check for valid config type
                if not isinstance(term_cfg, TerminationTermCfg):
                    raise TypeError(
                        f"Configuration for the term '{term_name}' is not of type TerminationTermCfg."
                        f" Received: '{type(term_cfg)}'."
                    )
                # resolve common parameters
                self._resolve_common_term_cfg(term_name, term_cfg, min_argc=1)
                # add function to list
                self._group_term_names[group_name].append(term_name)
                self._group_term_cfgs[group_name].append(term_cfg)
                # check if the term is a class
                if isinstance(term_cfg.func, ManagerTermBase):
                    self._group_class_term_cfgs[group_name].append(term_cfg)
