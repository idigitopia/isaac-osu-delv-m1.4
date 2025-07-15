# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

import wandb

# local imports
import cli_args  # isort: skip


def get_parser():
    # add argparse arguments
    parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
    parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
    parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
    parser.add_argument(
        "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
    )
    parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
    parser.add_argument("--task", type=str, default=None, help="Name of the task.")
    parser.add_argument(
        "--use_pretrained_checkpoint",
        action="store_true",
        help="Use the pre-trained checkpoint from Nucleus.",
    )
    parser.add_argument(
        "--wandb_checkpoint",
        type=str,
        default=None,
        help="Path to the checkpoint to load. Eg: 'team-osu/isaaclab/wpmvq8dk/model_1100.pt'",
    )
    parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
    parser.add_argument("--interactive", action="store_true", default=False, help="Run in interactive mode.")
    parser.add_argument("--interactive_mode", type=str, default="keyboard", help="Interactive mode.")

    parser.add_argument("--export_model_to_onnx", action="store_true", default=False, help="Export the model to onnx.")

    # append RSL-RL cli arguments
    cli_args.add_rsl_rl_args(parser)

    AppLauncher.add_app_launcher_args(parser)

    return parser


parser = get_parser()
# append AppLauncher cli args
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

import importlib
import os
import time
from datetime import datetime

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import torch
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_tasks.utils.hydra import hydra_task_config
from rsl_rl.runners import OnPolicyRunner

# import extensions to setup environments
import drail_extensions
from drail_extensions.core.isaaclab_rsl_rl.utils import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)


def preprocess_env_agent_cfg(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg
):
    """Preprocess the environment and agent configuration."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

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
            resume_path = agent_cfg.raw_checkpoint
        elif agent_cfg.wandb_checkpoint:
            import wandb

            run_path, file_name = agent_cfg.wandb_checkpoint.rsplit("/", 1)
            api = wandb.Api()
            run = api.run(run_path)
            file_ = run.file(file_name)
            file_.download(root=run_path, replace=True)
            resume_path = os.path.join(run_path, file_name)
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    else:
        resume_path = None

    return resume_path


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
    env = gym.wrappers.RecordVideo(env, **video_kwargs)
    return env


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
        runner.load(resume_path)

    return runner


def activate_interactive_mode_logic(args_cli: argparse.Namespace, env: gym.Env):
    """Activate the keyboard interactive logic."""

    if args_cli.interactive:
        if args_cli.interactive_mode == "keyboard":
            for term, command_obj in env.unwrapped.command_manager._terms.items():
                if hasattr(command_obj, "_activate_keyboard"):
                    command_obj._activate_keyboard = True
                    print(f"[INFO] Activated keyboard mode for {term} command.")
                else:
                    print(f"[INFO] No Keybaord Activations found for {term} command.")
    else:
        print("[INFO] Interactive mode is disabled.")


def export_model_to_onnx(ppo_runner: OnPolicyRunner, agent_cfg: RslRlOnPolicyRunnerCfg, log_root_path: str):
    """Export the model to onnx."""
    resume_path = get_resume_path(log_root_path, agent_cfg)
    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(
        ppo_runner.alg.actor_critic, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
    )
    export_policy_as_onnx(
        ppo_runner.alg.actor_critic, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )


def play_with_policy(policy: torch.nn.Module, env: gym.Env):
    dt = env.unwrapped.physics_dt

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, rewards, dones, infos = env.step(actions)

        if any(dones):
            print(f"[INFO] Episode finished with dones: {dones.nonzero()}")

        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time:
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                print(f"[INFO] Real-time mode is enabled. Sleeping for {sleep_time} seconds.")


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg, agent_cfg = preprocess_env_agent_cfg(env_cfg, agent_cfg)
    log_root_path, log_dir = get_log_utils(agent_cfg)

    # agent_cfg resume override.
    # agent_cfg.resume = True

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # wrap for video recording
    env = wrap_env_for_video(env, log_dir) if args_cli.video else env

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env) if not hasattr(env_cfg, "wrapper_class") else env_cfg.wrapper_class(env)

    # activate interactive mode
    activate_interactive_mode_logic(args_cli, env)

    # Get runner
    ppo_runner = setup_runner(env, agent_cfg, log_root_path, log_dir)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export model to onnx
    if args_cli.export_model_to_onnx:
        export_model_to_onnx(ppo_runner, agent_cfg, log_root_path)

    # play with the policy
    play_with_policy(policy, env)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
