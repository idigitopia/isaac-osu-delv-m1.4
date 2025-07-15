# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

#pose_wrapper.py

# Imports
import argparse
import importlib
import os
import sys
import time
import pandas as pd
import numpy as np
import gymnasium as gym
import torch
from isaaclab.app import AppLauncher

import wandb

# local imports
import cli_args  # isort: skip


def initialize_args_and_app_launcher():
    """
    Must be called first before calling anything else. because this sets the isaacsim enviornment dependencies. 
    """
    # add argparse arguments
    parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
    parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
    parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
    parser.add_argument(
        "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
    )
    parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
    parser.add_argument("--task", type=str, default=None, help="Name of the task.")
    parser.add_argument("--record_traj", action="store_true", default=False, help="Record trajectory during play.")
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
    # args_cli = parser.parse_args()
    # always enable cameras to record video
    if args_cli.video:
        args_cli.enable_cameras = True
    sys.argv = [sys.argv[0]] + hydra_args

    # launch omniverse app
    app_launcher = AppLauncher(args_cli)

    return args_cli, app_launcher


# INITALIZE APP LAUNCHER
args_cli, app_launcher = initialize_args_and_app_launcher()
simulation_app = app_launcher.app

# Secondary Imports for Isaacsim Dependencies, 
import isaaclab_tasks  # noqa: F401, E402
from isaaclab.envs import (  # noqa: E402
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402
from isaaclab.utils.io import dump_pickle # noqa: E402
from isaaclab.utils.pretrained_checkpoint import (  # noqa: E402
    get_published_pretrained_checkpoint,
)
import isaaclab.utils.math as math_utils  # noqa: E402
from drail_extensions.research_ashton.agents import (  # noqa: E402
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E401, E402

# import extensions to setup environments
import drail_extensions  # noqa: F401, E402

from drail_extensions.core.utils.keyboard import Keyboard  # noqa: E402





# **************************************************************************************
#########################################################################################
## Finite State Machine , BASE Code Block ##############################################
#########################################################################################


def direct_fsm_pose_wrapper_v2(robot_position, robot_heading, robot_quat, target_xyz_position):
    """Finite State Machine (FSM) wrapper for the pose command.
    Target pose is a 2D pose in the world frame (x, y, z, heading).
    Returns the velocity commands for the robot in the body frame.
    
    Output Format:
    vel_commands = torch.zeros((env.num_envs, 4), device=env.unwrapped.device)
    
    Matrix Structure:
    Column 0: x_velocity (linear velocity along x-axis)
    Column 1: y_velocity (linear velocity along y-axis)
    Column 2: z_angular_velocity (angular velocity around z-axis)
    Column 3: stand_bit (1 if target reached, 0 otherwise)
    
    Example Tensor:
    vel_commands = torch.tensor([
        [x_vel, y_vel, z_ang_vel, stand_bit],  # Environment 1
        [x_vel, y_vel, z_ang_vel, stand_bit],  # Environment 2
        [x_vel, y_vel, z_ang_vel, stand_bit],  # Environment 3
        [x_vel, y_vel, z_ang_vel, stand_bit]   # Environment 4
    ], device=env.unwrapped.device)
    """

    num_envs = 1
    # target_xyz_position = target_pose[:, :3]

    # Get the current pose and heading of the robot in world frame
    current_robot_xyz_position = robot_position
    current_heading = robot_heading # angle in world frame
    current_quat = robot_quat # quaternion in world frame

    # Calculate the position and heading commands in the body frame
    localized_target_position = math_utils.quat_rotate_inverse(math_utils.yaw_quat(current_quat), target_xyz_position - current_robot_xyz_position)
    dist_to_target = torch.linalg.norm(localized_target_position[:, :2], dim=1)
    target_heading_delta = math_utils.wrap_to_pi(0 - current_heading)

    # Check if the target has been reached
    target_reached_mask = (dist_to_target <= 0.1) & (target_heading_delta.abs() <= 0.1) # stand bit

    # Initialize and Set the velocity commands ###
    vel_commands = torch.zeros((num_envs, 4), device="cpu")
    # Clamp the vector delta to get linear velocity command.
    vel_commands[~target_reached_mask, :2] = torch.clamp(localized_target_position[~target_reached_mask, :2], -1.0, 1.0)
    # Clamp the heading delta to get angular velocity command.
    vel_commands[~target_reached_mask, 2] = torch.clamp(target_heading_delta[~target_reached_mask], -1.0, 1.0)
    # State 4: Stop (if reached target stop)
    vel_commands[target_reached_mask, 3] = 1.0  # Set the stand-bit to 1
    #################################################################################

    # Print the current pose and target pose
    # print(f"Target pose: {target_xyz_position} , 0 | Current pose: {current_robot_xyz_position[:, :2]}, {current_heading}", end="\r")
    # print(f"Vel commands: {vel_commands}")
    return vel_commands


def direct_fsm_pose_wrapper(depth_image, rgb_image, robot_position, robot_heading, robot_quat, target_pose):
    """Finite State Machine (FSM) wrapper for the pose command.
    Target pose is a 2D pose in the world frame (x, y, z, heading).
    Returns the velocity commands for the robot in the body frame.
    
    Output Format:
    vel_commands = torch.zeros((env.num_envs, 4), device=env.unwrapped.device)
    
    Matrix Structure:
    Column 0: x_velocity (linear velocity along x-axis)
    Column 1: y_velocity (linear velocity along y-axis)
    Column 2: z_angular_velocity (angular velocity around z-axis)
    Column 3: stand_bit (1 if target reached, 0 otherwise)
    
    Example Tensor:
    vel_commands = torch.tensor([
        [x_vel, y_vel, z_ang_vel, stand_bit],  # Environment 1
        [x_vel, y_vel, z_ang_vel, stand_bit],  # Environment 2
        [x_vel, y_vel, z_ang_vel, stand_bit],  # Environment 3
        [x_vel, y_vel, z_ang_vel, stand_bit]   # Environment 4
    ], device=env.unwrapped.device)
    """

    num_envs = depth_image.shape[0]
    target_xyz_position = target_pose[:, :3]

    # Get the current pose and heading of the robot in world frame
    current_robot_xyz_position = robot_position
    current_heading = robot_heading # angle in world frame
    current_quat = robot_quat # quaternion in world frame

    # Calculate the position and heading commands in the body frame
    localized_target_position = math_utils.quat_rotate_inverse(math_utils.yaw_quat(current_quat), target_xyz_position - current_robot_xyz_position)
    dist_to_target = torch.linalg.norm(localized_target_position[:, :2], dim=1)
    target_heading_delta = math_utils.wrap_to_pi(target_pose[:, 3] - current_heading)

    # Check if the target has been reached
    target_reached_mask = (dist_to_target <= 0.1) & (target_heading_delta.abs() <= 0.1) # stand bit

    # Initialize and Set the velocity commands ###
    vel_commands = torch.zeros((num_envs, 4), device=depth_image.device)
    # Clamp the vector delta to get linear velocity command.
    vel_commands[~target_reached_mask, :2] = torch.clamp(localized_target_position[~target_reached_mask, :2], -1.0, 1.0)
    # Clamp the heading delta to get angular velocity command.
    vel_commands[~target_reached_mask, 2] = torch.clamp(target_heading_delta[~target_reached_mask], -1.0, 1.0)
    # State 4: Stop (if reached target stop)
    vel_commands[target_reached_mask, 3] = 1.0  # Set the stand-bit to 1
    #################################################################################

    # Print the current pose and target pose
    print(f"Target pose: {target_pose} | Current pose: {current_robot_xyz_position[:, :2]}, {current_heading}", end="\r")

    return vel_commands


# Resample Target Pose ################################################################
def resample_target_pose(env):
    """Resample the target pose for all environments."""
    # obtain env origins for the environments
    target_pose = env.unwrapped.scene.env_origins.clone()

    r = torch.empty_like(target_pose[:, 0], device=env.unwrapped.device)
    target_pose[:, 0] += r.uniform_(-3.0, 3.0)  # x position
    target_pose[:, 1] += r.uniform_(-3.0, 3.0)  # y position
    target_pose[:, 2] = env.unwrapped.scene["robot"].data.default_root_state.clone()[:, 2]  # z position
    heading = r.uniform_(-torch.pi, torch.pi)  # heading command

    return torch.cat([target_pose, heading.unsqueeze(1)], dim=1)

#########################################################################################

### FSM CLASS ###
class LanguageGroundedFSM:
    """
    FSM that navigates to a target object with a natural language command.
    """
    def __init__(self, device):
        self.device = device
        self.state = "SCANNING"  # Start directly in navigating state
        self.scan_start_heading = None
        self.scan_images = {}
        self.previous_heading = None  
        self.cumulative_rotation = None 
        self.last_capture_rotation = None 

        self.navigation_instructions = []  # x, y, z, heading
        self.instruction_completed = False
        self.target_object_id = "target_coordinates"
        self.object_navigation_path = None
        self.instruction_generated = False

        # Add these new variables to store the navigation instruction
        self.navigation_instruction = None
        self.current_target_xyz_position = None
        self.user_command_processed = False  # Flag to track if user input was already processed
        
    def update(self, depth_image, rgb_image, robot_position, robot_heading, robot_quat, target_pose, camera_params):
        num_envs = depth_image.shape[0]
        vel_commands = torch.zeros((num_envs, 4), device=self.device)
        
        if self.state == "SCANNING":
            return self._handle_scanning(vel_commands, robot_heading, depth_image, rgb_image, robot_position, robot_quat, camera_params)
        elif self.state == "PROCESSING":
            return self._handle_processing(vel_commands, robot_position, robot_heading, robot_quat)
        elif self.state == "NAVIGATING":
            return self._handle_navigating(vel_commands, robot_position, robot_heading, robot_quat)
        elif self.state == "COMPLETED":
            return self._handle_completed(vel_commands)
        else:
            vel_commands[:, 3] = 1.0
            return vel_commands
                    
    def _handle_scanning(self, vel_commands, robot_heading, depth_image, rgb_image, robot_position, robot_quat, camera_params):
        """
        Robot does a 360 degree turn while scanning environment for RGBD images every 25 degrees.
        Stores metadata in self.scan_images dict.
        """
        if self.scan_start_heading is None:
            self.scan_start_heading = robot_heading.clone()
            self.previous_heading = robot_heading.clone()
            self.cumulative_rotation = torch.zeros_like(robot_heading)
            self.last_capture_rotation = torch.zeros_like(robot_heading)
            print(f"[FSM] Starting scan at heading: {self.scan_start_heading}")
        
        
        vel_commands[:, 2] = 1.0 # rads/s angular velocity
        
        current_heading = robot_heading
        heading_delta = current_heading - self.previous_heading
        heading_delta = torch.atan2(torch.sin(heading_delta), torch.cos(heading_delta))
        self.cumulative_rotation += torch.abs(heading_delta)
        self.previous_heading = current_heading.clone()
        
        # Capture images every ~25* (π/7 radians)
        rotation_since_last_capture = self.cumulative_rotation - self.last_capture_rotation
        if rotation_since_last_capture.item() >= (torch.pi / 7):
            # Construct full 7D pose tensor: [x, y, z, qw, qx, qy, qz]
            robot_pose = torch.cat([robot_position, robot_quat], dim=1)  # Shape: (num_envs, 7)
            
            scan_data = {
                'depth': depth_image.clone(),
                'rgb': rgb_image.clone(),
                'rotation_degrees': torch.rad2deg(self.cumulative_rotation).item(),
                'image_index': len(self.scan_images),
                'robot_pose': robot_pose.clone(),  # Save full 7D pose instead of just position
                'camera_params': camera_params
            }
            self.scan_images[len(self.scan_images)] = scan_data
            self.last_capture_rotation = self.cumulative_rotation.clone()
            print(f"[FSM] Captured image #{len(self.scan_images)} at {torch.rad2deg(self.cumulative_rotation).item():.1f}°")
        
        # Check if 360* rotation is completed
        rotation_completed = self.cumulative_rotation >= (2 * torch.pi - 0.1)
        
        print(f"[FSM] Cumulative rotation: {self.cumulative_rotation.item():.2f} rad ({torch.rad2deg(self.cumulative_rotation).item():.1f}°)")
        

        if rotation_completed.all():
            print(f"[FSM] Scan completed: 360* rotation finished with {len(self.scan_images)} images")
            self.state = "PROCESSING"
            vel_commands[:, 2] = 0.0  # Stop rotation
            vel_commands[:, 3] = 1.0  # Stand bit
            self._save_scan_data()  
            
        return vel_commands
    
    def _save_scan_data(self):
        """Save all scan data to files"""
        print(f"LAST SCANNED METADATA: {self.scan_images[len(self.scan_images)-1]}")
        print("SAVING ALL METADATA TO CSV:")
        
        import time
        save_dir = f"tiamat_fsm_task_scripts/data/scan_data"
        os.makedirs(save_dir, exist_ok=True) 
        os.makedirs(save_dir+"/rgb/", exist_ok=True)
        os.makedirs(save_dir+"/depth/", exist_ok=True) 
        os.makedirs(save_dir+"/pose/", exist_ok=True)  # Changed from position to pose
        
        metadata_list = []
        
        for img_idx, scan_data in self.scan_images.items():
            # Save RGB and depth images as .pt files
                ##NOTE: potentially save as .pngs
            rgb_path = f"{save_dir}/rgb/rgb_{img_idx:03d}.pt"
            depth_path = f"{save_dir}/depth/depth_{img_idx:03d}.pt"
            pose_path = f"{save_dir}/pose/pose_{img_idx:03d}.pt"  # Changed from position to pose
            camera_path = f"{save_dir}/camera/camera_{img_idx:03d}.pt"
            
            torch.save(scan_data['rgb'], rgb_path)
            torch.save(scan_data['depth'], depth_path)
            torch.save(scan_data['robot_pose'], pose_path)  # Save full 7D pose tensor
            # torch.save(scan_data['camera_params'], camera_path)
            
            # Collect metadata
            metadata_list.append({
                'image_index': img_idx,
                'rotation_degrees': scan_data['rotation_degrees'],
                'rgb_file': rgb_path,
                'depth_file': depth_path,
                'pose_file': pose_path,  # Changed from position_file to pose_file
                # 'camera_intrinsics_file': camera_path
            })
        
        
        df = pd.DataFrame(metadata_list)
        csv_path = f"{save_dir}/scan_metadata.csv"
        df.to_csv(csv_path, index=False)
        
        print(f"[SAVE] Saved {len(self.scan_images)} images and metadata to: {save_dir}/")
        print(f"[SAVE] CSV file: {csv_path}")
        
    def _handle_processing(self, vel_commands, robot_position, robot_heading, robot_quat, ):
        """Handle the processing state - robot stands still while processing images"""
        vel_commands[:, 3] = 1.0  # Stand bit
        print("[FSM] Processing images... Robot standing still.")
        
        try:
            if not self.instruction_generated:
                results, filepath = self.process_images()
                from VLM_preds import VLM_preds

                filepath = VLM_preds()

                
                descriptions_df = pd.read_csv(filepath)

                
                for idx, desc in enumerate(descriptions_df['object_descriptions']):
                    print(f"Image {descriptions_df.loc[idx, 'image_index']}: {desc}")
                
                ##NOTE: At this stage (assuming the navigation coordinates have been fixed), we should be at a good point to 
                    # incorporate an LLM to navigate the robot.
                

                print("="*60)
                print("GENERATING NAVIGATION INSTRUCTIONS")
                print("="*60)

                from map import navigation
                self.navigation_instructions, self.object_navigation_path = navigation(filepath)
                
                #self.navigation_instructions = all_instructions
                #print("NAVIGATION INSTRUCTIONS: ", self.navigation_instructions)
                
            

                
                print("NAVIGATION INSTRUCTIONS GENERATED.")
                print("="*60)
                self.instruction_generated = True
        except Exception as e:
            # This saves a lot of headaches with the process freezing due to vel_cmds override
            print(f"[FSM] Error during processing: {e}")
            print("[FSM] Transitioning to COMPLETED state due to processing error.")
            self.state = "COMPLETED"
            return vel_commands

        if self.instruction_generated:
            # get natural language command from user
            from language_instruction import get_navigation_instruction
            natural_language_command = input("Where would you like the robot to navigate? Enter natural language command: ")
            self.navigation_instruction = get_navigation_instruction(json_filepath=self.object_navigation_path, command=natural_language_command)
            self.current_target_xyz_position = self.navigation_instruction[:3]


            print("="*60)
            print(f"API RETURNED NAVIGATION INSTRUCTION. TRANSITIONING TO NAVIGATING STATE.")
            print("="*60)
            self.state = "NAVIGATING"
            if natural_language_command == "scan world again":
                self.state = "SCANNING"
            return self._handle_navigating(vel_commands, robot_position, robot_heading, robot_quat)

    def process_images(self):
        """
        Process the captured images
        """
        from object_detection import detection
        print(f"[FSM] Processing {len(self.scan_images)} captured images...")
        
        results, filepath = detection(csv_file_path='tiamat_fsm_task_scripts/data/scan_data/scan_metadata.csv', 
         output_dir="tiamat_fsm_task_scripts/data/bounded_imgs", 
         max_width=300, max_height=300, distance_threshold=50, 
         saturation_threshold=30, brightness_diff_threshold=40)
        
        print("[FSM] Image processing completed.")
        return results, filepath

    def _handle_navigating(self, vel_commands, robot_position, robot_heading, robot_quat):
        """Navigate to the stored navigation instruction"""
        
        
        
        
        # Get navigation commands using stored target
        vel_commands = direct_fsm_pose_wrapper_v2(robot_position, robot_heading, robot_quat, torch.FloatTensor([self.current_target_xyz_position]))
        
        current_pos = robot_position[0]  # Single environment
        target_pos = torch.tensor(self.current_target_xyz_position, device=self.device)
        distance_to_target = torch.linalg.norm(current_pos[:2] - target_pos[:2])
        
        # Display progress
        print(f"[NAVIGATING] Target: ({self.current_target_xyz_position[0]:.2f}, {self.current_target_xyz_position[1]:.2f}, {self.current_target_xyz_position[2]:.2f}) | Distance: {distance_to_target:.2f}m", end="\r")
        
        # Check if reached target
        if distance_to_target <= 0.18:
            print(f"\n[FSM] Successfully reached target")
            print("[FSM] Transitioning to COMPLETED state.")
            self.state = "COMPLETED"
            vel_commands[:, 3] = 1.0  # Stand bit set
        
        return vel_commands
        
    def _handle_completed(self, vel_commands):
        """Handle the completed state - robot stands still"""
        vel_commands[:, 3] = 1.0  # Stand bit
        print("[FSM] Navigation completed. Robot standing still.", end="\r")

        # Go prompt user for a new command
        self.state = "PROCESSING"
        return vel_commands

    def reset(self):
        """Reset the FSM to initial state"""
        self.state = "NAVIGATING"  # Reset to navigating state
        self.scan_start_heading = None
        self.scan_images = {}
        self.previous_heading = None
        self.cumulative_rotation = None
        self.last_capture_rotation = None
        self.instruction_completed = False
        self.target_object_id = "target_coordinates"
        print("[FSM] FSM reset to navigating state.")

        # Reset navigation-related variables
        self.navigation_instruction = None
        self.current_target_xyz_position = None
        self.user_command_processed = False
# **************************************************************************************



@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""


    #########################################################################################
    ## Get Configurations ###################################################################
    #########################################################################################

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    #########################################################################################



    #########################################################################################
    ## Set logging Parameters / Checkpoint ##################################################
    #########################################################################################

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", agent_cfg.experiment_name)
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
    #########################################################################################




    #########################################################################################
    ## Create Environment ###################################################################
    #########################################################################################

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
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    #########################################################################################



    #########################################################################################
    ## Load Checkpoint #####################################################################
    #########################################################################################

    # Get runner_class with fallback to default value
    runner_class_path = getattr(agent_cfg, "class_name", "rsl_rl.runners.OnPolicyRunner")
    module_path, class_name = runner_class_path.rsplit(".", 1)
    runner_class = getattr(importlib.import_module(module_path), class_name)
    print(f"[INFO] Runner class: {runner_class}")

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = runner_class(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path, load_optimizer=False)
    print(
        f"[INFO] Loaded model checkpoint from: {resume_path}. Learning iteration:"
        f" {ppo_runner.current_learning_iteration}"
    )

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")


    dt = env.unwrapped.step_dt
    

    # Keyboard Input #######################################################################
    keyboard_input = Keyboard()
    key_cooldown = {"P": False}

    # reset environment
    obs, infos = env.get_observations()
    timestep = 0


    # **********************************************************************************
    #################################################################################
    ### FSM CODE BLOCK 1 ############################################################
    ## Use this to initialize anything at the beginning of the play script.
    ## This is executed only once at the beginning of the play script.
    #################################################################################
    
    target_pose = resample_target_pose(env)
    
    # Initialize FSM instance ONCE
    fsm_instance = LanguageGroundedFSM(env.unwrapped.device)
    language_command = "go to the red box"

    #################################################################################
    # **********************************************************************************

    time_step_count = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():

            # camera_obs = env.unwrapped.scene["robot"].data.camera_obs.clone()
            depth_image = infos['observations']['camera']['depth_image']
            rgb_image = infos['observations']['camera']['rgb_image']


            robot_position = env.unwrapped.scene["robot"].data.root_pos_w.clone()
            robot_heading = env.unwrapped.scene["robot"].data.heading_w.clone()
            robot_quat = env.unwrapped.scene["robot"].data.root_quat_w.clone()

            camera_pos = env.unwrapped.scene["rgbd_camera"].data.pos_w.clone()
            camera_quat = env.unwrapped.scene["rgbd_camera"].data.quat_w_ros.clone()

            # **********************************************************************************
            #################################################################################
            ### FSM CODE BLOCK 2 ############################################################
            ## This is executed every step.
            ## This is where you get observations and use them to compute the velocity commands. 
            ## FSM gets access to the environment camera image and other observations.
            #################################################################################

            # get fsm velocity commands using the EXISTING instance
            #vel_cmds = direct_fsm_pose_wrapper(depth_image, rgb_image, robot_position, robot_heading, robot_quat, target_pose)
            camera_params = {"camera_pos": env.unwrapped.scene["rgbd_camera"].data.pos_w.clone(), 
                                "camera_quat": env.unwrapped.scene["rgbd_camera"].data.quat_w_world.clone(), 
                                "camera_quat_opengl": env.unwrapped.scene["rgbd_camera"].data.quat_w_opengl.clone(),
                                "camera_quat_w_ros": env.unwrapped.scene["rgbd_camera"].data.quat_w_ros.clone(),
                                 }
            if time_step_count > 50*10:# wait for env to settle
                if fsm_instance.state == "NAVIGATING":
                    # use robot position, robot quaternion when navigating.
                    vel_cmds = fsm_instance.update(depth_image, rgb_image, robot_position, robot_heading, robot_quat, target_pose, camera_params)
                else:
                    vel_cmds = fsm_instance.update(depth_image, rgb_image, camera_pos, robot_heading, camera_quat, target_pose, camera_params)
            else:
                vel_cmds = direct_fsm_pose_wrapper_v2(robot_position, robot_heading, robot_quat, torch.FloatTensor([0,0,0]))
            time_step_count += 1
            #Sanity check
            # import numpy as np
            # current_target_xyz_position = np.array([6,6,0])
            # vel_cmds = direct_fsm_pose_wrapper_v2(robot_position, robot_heading, robot_quat, torch.FloatTensor([current_target_xyz_position[:3]]))
            ## NOTE: This is expecting that policy has the last 4 dimensions for velocity commands.
            ## The velocity commands are in the order (x_vel, y_vel, ang_vel, stand_bit).
            ## Change this if the policy expects a different order.

            #################################################################################
            # **********************************************************************************

            # override the observations with the velocity commands
            obs[:, -4:] = vel_cmds
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, infos = env.step(actions)

            # check for keyboard input #####################################################
            key = keyboard_input.get_input()
            if key is not None:
                if key == "R":
                    # reset the environment
                    _, _ = env.reset()
                    obs, infos = env.get_observations()
                    # Reset FSM state when environment resets
                    fsm_instance.state = "NAVIGATING"
                    fsm_instance.scan_start_heading = None
                    fsm_instance.scan_images = {}
                elif key == "ESCAPE":
                    # exit the play loop
                    break
            else:
                # reset the cooldown for the key
                key_cooldown["P"] = False
            #################################################################################


        ## Video Recording ###################################################################
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break
        #################################################################################

        # time delay for real-time evaluation #############################################
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)
        #################################################################################

        # resample target pose every 1000 steps #########################################
        timestep += 1


        # **********************************************************************************
        #################################################################################
        ### FSM CODE BLOCK 3 ############################################################
        ## This is executed every 1000 steps. change the number of steps to whatever you want.
        ## This is where you can resample the target pose for example.
        ## This is where you can do any other periodic tasks.
        #################################################################################
        # if timestep % 1000 == 0:
        #     print(f"[INFO] Resampling target pose at timestep {timestep}.")
        #     target_pose = resample_target_pose(env)
        #################################################################################
        # **********************************************************************************


    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()