from isaaclab.utils import configclass


@configclass
class RewardGroupCfg:
    """Configuration for a reward group.

    Reward groups can be used to compute sets of reward terms independently.
    This can be useful to compute separate rewards in multi-agent scenarios.
    """


@configclass
class TerminationGroupCfg:
    """Configuration for a termination group.

    Termination groups can be used to compute sets of termination terms independently.
    This can be useful to compute separate terminations in multi-agent scenarios.
    """
