from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass

# from drail_extensions.core.isaaclab_rsl_rl.utils import *
import isaaclab_rl.rsl_rl.rl_cfg as base_rsl_rl_cfg


@configclass
class RslRlPpoActorCriticCfg(base_rsl_rl_cfg.RslRlPpoActorCriticCfg):
    """Configuration for the PPO actor-critic networks."""

    learnable_std: bool = True
    """Whether to make the std learnable. Default is True."""


@configclass
class RslRlPpoAlgorithmCfg(base_rsl_rl_cfg.RslRlPpoAlgorithmCfg):
    """Configuration for the PPO algorithm."""

    lr_scheduler: bool = False
    """Whether to use the step-based optimizer lr scheduler or KL-based lr scheduler."""

    lr_scheduler_gamma: float = MISSING
    """The decay rate for the optimizer scheduler."""

    lr_scheduler_step_size: int = MISSING
    """The step size for the optimizer scheduler."""


@configclass
class RslRlOnPolicyRunnerCfg(base_rsl_rl_cfg.RslRlOnPolicyRunnerCfg):
    """Configuration of the runner for on-policy algorithms."""

    wandb_checkpoint: str = ""
    """The wandb checkpoint to load. Default is empty string."""

    raw_checkpoint: str = ""
    """The raw checkpoint to load. Default is empty string."""

    clip_actions: float | None = None
    """The clipping value for actions. If ``None``, then no clipping is done."""

    wandb_group: str = ""
    """The wandb group name. Default is empty string."""

    wandb_entity: str = ""
    """The wandb entity name. Default is empty string."""

    load_optimizer: bool = True
    """Whether to load the optimizer state. Default is True.

    If True, the optimizer state is loaded. If False, only the model is loaded.
    """


@configclass
class RslRlPpoActorCriticDepthMapCfg(RslRlPpoActorCriticCfg):
    """Configuration for the PPO actor-critic networks."""

    depth_map_encoded_dim: int = 256
    """The dimension of the depth map encoder."""


@configclass
class RslRlPpoActorCriticPointCloudCfg(RslRlPpoActorCriticCfg):
    """Configuration for the PPO actor-critic networks."""

    point_cloud_encoded_dim: int = 256
    """The dimension of the point cloud encoder."""

    point_cloud_downsample_factor: int = 4
    """The downsample factor for the point cloud encoder."""

    point_cloud_downsample_steps: int = 2
    """The number of downsample steps for the point cloud encoder."""


@configclass
class RslRlPpoActorCriticRecurrentCfg(RslRlPpoActorCriticCfg):
    """Configuration for the PPO actor-critic networks."""

    class_name: str = "ActorCriticRecurrent"
    """The policy class name. Default is ActorCritic."""

    rnn_type: str = "lstm"
    """The type of RNN to use. Default is lstm."""

    rnn_hidden_size: int = 128
    """The hidden size of the RNN."""

    rnn_num_layers: int = 2
    """The number of layers in the RNN."""


@configclass
class RslRlPpoActorCriticDepthMapRecurrentCfg(RslRlPpoActorCriticDepthMapCfg, RslRlPpoActorCriticRecurrentCfg):
    """Configuration combining depth map and recurrent actor-critic networks."""

    pass


@configclass
class RslRlPpoAlgorithmPointCloudCfg(RslRlPpoAlgorithmCfg):
    """Configuration for the PPO algorithm."""

    point_cloud_encoder_learning_freq: int = 1
    """The frequency (in term of epochs) of learning the point cloud encoder in ppo update loop. This is too
    speed up the training of the point cloud encoder.
    """
