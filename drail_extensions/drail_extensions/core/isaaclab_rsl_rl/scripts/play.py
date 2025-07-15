# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import argparse
import importlib
import os
import sys
import time

import gymnasium as gym
import torch
from isaaclab.app import AppLauncher

import wandb

# local imports
import cli_args  # isort: skip

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
parser.add_argument("--export_policy", action="store_true", default=False, help="Export policy to onnx/jit.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import traceback  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import omni.log  # noqa: E402
from isaaclab.envs import (  # noqa: E402
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
)
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402
from isaaclab.utils.pretrained_checkpoint import (  # noqa: E402
    get_published_pretrained_checkpoint,
)
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E401, E402

# import extensions to setup environments
# import drail_extensions  # noqa: F401, E402
from drail_extensions.core.isaaclab_rsl_rl.utils import (  # noqa: E402
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    elif args_cli.wandb_checkpoint:
        download_path_prefix = os.path.join("logs", "rsl_rl", "play", args_cli.task)
        os.makedirs(download_path_prefix, exist_ok=True)
        run_path, file_name = args_cli.wandb_checkpoint.rsplit("/", 1)
        api = wandb.Api()
        run = api.run(run_path)
        file_ = run.file(file_name)
        download_path = os.path.join(download_path_prefix, run_path)
        file_.download(root=download_path, replace=True)
        resume_path = os.path.join(download_path, file_name)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    if hasattr(env_cfg, "wrapper_class"):
        env = env_cfg.wrapper_class(env)
    else:
        env = RslRlVecEnvWrapper(env)

    # Get runner_class with fallback to default value
    runner_class_path = getattr(agent_cfg, "class_name", "rsl_rl.runners.OnPolicyRunner")
    module_path, class_name = runner_class_path.rsplit(".", 1)
    runner_class = getattr(importlib.import_module(module_path), class_name)
    print(f"[INFO] Runner class: {runner_class}")

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = runner_class(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    model_loaded = False
    try:
        ppo_runner.load(resume_path)
        print(
            f"[INFO] Loaded model checkpoint from: {resume_path}. Learning iteration:"
            f" {ppo_runner.current_learning_iteration}"
        )
        model_loaded = True
    except Exception as e:
        omni.log.error(f"Failed to load model checkpoint from: {resume_path}. Error: {e}")
        omni.log.error(traceback.format_exc())

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    if args_cli.export_policy:
        # export policy to onnx/jit
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        export_policy_as_jit(
            ppo_runner.alg.actor_critic, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
        )
        export_policy_as_onnx(
            ppo_runner.alg.actor_critic,
            # input_dims={
            #     "observations": env.observation_space["policy"].shape[1:],
            #     "depth_map": env.observation_space["depth_map"].shape[1:],
            # },
            normalizer=ppo_runner.obs_normalizer,
            path=export_model_dir,
            filename="policy.onnx",
        )

    dt = env.unwrapped.physics_dt

    if not hasattr(agent_cfg, "class_name"):
        agent_cfg.class_name = "rsl_rl.runners.OnPolicyRunner"

    # reset environment
    obs, infos = env.get_observations()
    # obs = obs.view(-1, 46)
    timestep = 0

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            if agent_cfg.class_name == "rsl_rl_point_clouds.runners.OnPolicyRunner":
                actions = policy(obs, point_cloud=infos["observations"]["point_cloud"])
            elif agent_cfg.class_name == "rsl_rl_depth_map.runners.OnPolicyRunner":
                actions = policy(obs, depth_map=infos["observations"]["depth_map"])
            else:
                actions = policy(obs)

            if not model_loaded:
                omni.log.error("Model weights not loaded.")

            obs, _, _, infos = env.step(actions)

        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
