from isaaclab.utils import configclass

from drail_extensions.core.isaaclab_tasks.tasks.SaW.go2 import rsl_rl_ppo_cfg


@configclass
class Go2_Reach_SE2_Command_PPORunnerCfg(rsl_rl_ppo_cfg.Go2FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.num_steps_per_env = 8
        self.experiment_name = "go2_reach_se2_command"

        self.policy.init_noise_std = 0.5
        self.policy.actor_hidden_dims = [128, 128]
        self.policy.critic_hidden_dims = [128, 128]

        self.algorithm.entropy_coef = 0.005
