from isaaclab.utils import configclass

from drail_extensions.core.isaaclab_tasks.tasks.navigation.go2_reach_se2_command.rsl_rl_reach_se2_command_ppo_cfg import (  # noqa: E501
    Go2_Reach_SE2_Command_PPORunnerCfg,
)


@configclass
class Go2_Reach_SE2_Box_PPORunnerCfg(Go2_Reach_SE2_Command_PPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "go2_reach_se2_box"
