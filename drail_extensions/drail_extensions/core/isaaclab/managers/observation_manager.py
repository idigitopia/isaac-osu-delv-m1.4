from isaaclab.managers import observation_manager

import drail_extensions.core.utils.string as string_utils


class ObservationManager(observation_manager.ObservationManager):
    def __str__(self) -> str:
        msg = super().__str__()
        msg += string_utils.get_common_terms_repr(self._group_obs_term_names, self._group_obs_term_cfgs)
        return msg
