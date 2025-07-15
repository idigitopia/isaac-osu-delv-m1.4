# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse  # noqa: E402
import sys  # noqa: E402

from isaaclab.app import AppLauncher  # noqa: E402

# local imports
import cli_args  # isort: skip # noqa: E402


def get_parser():
    # add argparse arguments
    parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
    parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
    parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
    parser.add_argument(
        "--video_interval", type=int, default=2000, help="Interval between video recordings (in steps)."
    )
    parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
    parser.add_argument("--task", type=str, default=None, help="Name of the task.")
    parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
    parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
    parser.add_argument(
        "--wandb_checkpoint",
        type=str,
        default=None,
        help="Path to the checkpoint to resume from. Eg: 'team-osu/isaaclab/wpmvq8dk/model_1100.pt'",
    )
    parser.add_argument(
        "--raw_checkpoint",
        type=str,
        default=None,
        help="Path to the checkpoint to resume from. Eg: 'team-osu/isaaclab/wpmvq8dk/model_1100.pt'",
    )
    # append RSL-RL cli arguments
    cli_args.add_rsl_rl_args(parser)
    # append AppLauncher cli args
    AppLauncher.add_app_launcher_args(parser)

    return parser


parser = get_parser()
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import importlib  # noqa: E402
import os  # noqa: E402
from datetime import datetime  # noqa: E402

import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: F401, E402
import torch  # noqa: E402
from isaaclab.envs import (  # noqa: E402
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
)
from isaaclab.utils.dict import print_dict  # noqa: E402
from isaaclab.utils.io import dump_pickle, dump_yaml  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

# import extensions to setup environments
import drail_extensions  # noqa: F401, E402
from drail_extensions.core.isaaclab_rsl_rl.utils import (  # noqa: E402
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def preprocess_env_agent_cfg(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg | DirectMARLEnv, agent_cfg: RslRlOnPolicyRunnerCfg
):
    """Preprocess the environment and agent configuration."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    return env_cfg, agent_cfg


def get_log_utils(agent_cfg: RslRlOnPolicyRunnerCfg):
    """Get the log utilities."""
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # This way, the Ray Tune workflow can extract experiment name.
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    return log_root_path, log_dir


def get_resume_path(log_root_path: str, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Get the resume path."""
    if agent_cfg.resume:
        if agent_cfg.raw_checkpoint:
            return agent_cfg.raw_checkpoint
        if agent_cfg.wandb_checkpoint:
            import wandb

            run_path, file_name = agent_cfg.wandb_checkpoint.rsplit("/", 1)
            api = wandb.Api()
            run = api.run(run_path)
            file_ = run.file(file_name)
            file_.download(root=run_path, replace=True)
            return os.path.join(run_path, file_name)
        return get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    return None


def wrap_env_for_video(env: gym.Env, log_dir: str):
    """Wrap the environment for video recording."""
    video_kwargs = {
        "video_folder": os.path.join(log_dir, "videos", "train"),
        "step_trigger": lambda step: step % args_cli.video_interval == 0,
        "video_length": args_cli.video_length,
        "disable_logger": True,
    }
    print("[INFO] Recording videos during training.")
    print_dict(video_kwargs, nesting=4)
    return gym.wrappers.RecordVideo(env, **video_kwargs)


def setup_runner(env: gym.Env, agent_cfg: RslRlOnPolicyRunnerCfg, log_root_path: str, log_dir: str):
    """Setup the runner."""
    runner_class_path = getattr(agent_cfg, "class_name", "rsl_rl.runners.OnPolicyRunner")
    module_path, class_name = runner_class_path.rsplit(".", 1)
    runner_class = getattr(importlib.import_module(module_path), class_name)
    print(f"[INFO] Runner class: {runner_class}")

    runner = runner_class(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume:
        resume_path = get_resume_path(log_root_path, agent_cfg)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        try:
            runner.load(resume_path, itr=0)
        except Exception as e:  # noqa: F841
            runner.load(resume_path)
            print(f"[ERROR]: Failed to load model checkpoint from with itr 0: {resume_path}")

    return runner


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""

    env_cfg, agent_cfg = preprocess_env_agent_cfg(env_cfg, agent_cfg)
    log_root_path, log_dir = get_log_utils(agent_cfg)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # wrap for video recording
    env = wrap_env_for_video(env, log_dir) if args_cli.video else env

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env) if not hasattr(env_cfg, "wrapper_class") else env_cfg.wrapper_class(env)

    # Get runner_class with fallback to default value
    runner = setup_runner(env, agent_cfg, log_root_path, log_dir)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training
    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations,
        init_at_random_ep_len=agent_cfg.init_at_random_ep_len if hasattr(agent_cfg, "init_at_random_ep_len") else False,
    )

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
