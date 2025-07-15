import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from go2_robot import Robot
from onnx_policy import OnnxPolicy
from utils import remap


class Runner:
    def __init__(self, policy_path, network, no_log):
        self.robot = Robot(network)

        self.policy = OnnxPolicy(policy_path)
        env_cfg_file = Path(policy_path).parent / "env.yaml"
        if os.path.exists(env_cfg_file):
            with open(Path(policy_path).parent / "env.yaml") as handle:
                env_cfg = yaml.load(handle, Loader=yaml.UnsafeLoader)
            # TODO: Ok to hardcode 'joint_pos' action name? Or just assume that there will be only 1 action?
            self.action_scale = env_cfg["actions"]["joint_pos"]["scale"]
            self.control_dt = env_cfg["decimation"] * env_cfg["sim"]["dt"]
        else:
            self.action_scale = 0.25
            self.control_dt = 1 / 50  # 50 Hz

        self.actions = np.zeros(self.robot.num_joints)

        self.no_log = no_log
        self.run_policy = False
        self.dt_threshold = 0.001
        self.pol_tick = 0

        # Joint IDs
        self._joint_names = [
            "FR_hip",  # Front right hip
            "FR_thigh",  # Front right thigh
            "FR_calf",  # Front right calf
            "FL_hip",  # Front left hip
            "FL_thigh",  # Front left thigh
            "FL_calf",  # Front left calf
            "RR_hip",  # Rear right hip
            "RR_thigh",  # Rear right thigh
            "RR_calf",  # Rear right calf
            "RL_hip",  # Rear left hip
            "RL_thigh",  # Rear left thigh
            "RL_calf",  # Rear left calf
        ]

        # Joint offsets
        self._joint_offsets = np.array([-0.1, 0.8, -1.5, 0.1, 0.8, -1.5, -0.1, 1.0, -1.5, 0.1, 1.0, -1.5])

        # Joint limits
        with open(
            os.path.join(os.path.dirname(os.path.realpath(__file__)), "joint_limits_go2/joint_limits_processed.yaml"),
        ) as file:
            joint_limits = yaml.safe_load(file)

        self._joint_limits_min = np.array(joint_limits["joint_limit_min"])
        self._joint_limits_max = np.array(joint_limits["joint_limit_max"])

        if not no_log:
            log_root_path = os.path.join("logs", "deployment")
            log_root_path = os.path.abspath(log_root_path)
            self.log_dir = os.path.join(
                log_root_path, os.path.basename(policy_path), datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            )
            print(f"Logging data to {self.log_dir}")

            self.max_log_size = 50000
            obs_size = self._get_observation().shape[0]  # 45
            cmd_size = self.commands.shape[0]
            obs_name = (
                [
                    "ang_vel_x",
                    "ang_vel_y",
                    "ang_vel_z",
                    "proj_grav_x",
                    "proj_grav_y",
                    "proj_grav_z",
                    "cmd_x",
                    "cmd_y",
                    "cmd_yaw",
                ]
                + [f"joint_pos_rel_{jnt_name}" for jnt_name in self._joint_names]
                + [f"joint_vel_{jnt_name}" for jnt_name in self._joint_names]
                + [f"action_{jnt_name}" for jnt_name in self._joint_names]
            )
            self.log_dict = {
                "low_freq": {
                    "observation": np.zeros((self.max_log_size, obs_size)),
                    "action": np.zeros((self.max_log_size, self.actions.shape[0])),
                    "commands": np.zeros((self.max_log_size, cmd_size)),
                    "timestamp": np.zeros(self.max_log_size),
                    "tick": np.zeros(self.max_log_size),
                    "obs_names": obs_name,
                    "action_names": self._joint_names,
                },
                "high_freq": {
                    "joint_position": np.zeros((self.max_log_size, self.robot.num_joints)),
                    "joint_velocity": np.zeros((self.max_log_size, self.robot.num_joints)),
                    "joint_torque": np.zeros((self.max_log_size, self.robot.num_joints)),
                    "imu_quaternion": np.zeros((self.max_log_size, 4)),
                    "imu_gyroscope": np.zeros((self.max_log_size, 3)),
                    "imu_accelerometer": np.zeros((self.max_log_size, 3)),
                    "foot_force": np.zeros((self.max_log_size, 4)),
                    "timestamp": np.zeros(self.max_log_size),
                    "tick": np.zeros(self.max_log_size),
                    "joint_order": self._joint_names,
                },
                "policy": policy_path,
            }
            self.curr_log_ind = 0
            self.curr_lowstate_log_ind = 0
            self.save_ind = 0
            self.curr_obs = np.zeros(obs_size)

    @property
    def last_action(self):
        return self.actions

    @property
    def joint_pos_rel(self):
        return [self.robot.joint_pos[i] - self._joint_offsets[i] for i in range(self.robot.num_joints)]

    @property
    def commands(self):
        # Clip joystick values to [-1, 1]
        x_vel = np.clip(self.robot.controller.Ly, -1.0, 1.0)
        # Flip x-axis values
        y_vel = np.clip(self.robot.controller.Lx, -1.0, 1.0) * -0.5
        yaw_vel = np.clip(self.robot.controller.Rx, -1.0, 1.0) * -1.0

        # Remap joystick values to training range
        x_vel = remap(x_vel, -1.0, 1.0, -1.0, 1.0)
        y_vel = remap(y_vel, -1.0, 1.0, -1.0, 1.0)
        yaw_vel = remap(yaw_vel, -1.0, 1.0, -1.0, 1.0)

        return np.array([x_vel, y_vel, yaw_vel])

    def _get_observation(self):
        # Need to cast as float32 since torch defaults to float32
        return np.concatenate(
            [
                self.robot.angular_velocity,
                self.robot.projected_gravity,
                self.commands,
                self.joint_pos_rel,
                self.robot.joint_vel,
                self.last_action,
            ],
            dtype=np.float32,
        )

    def _set_target_joint_positions(self):
        self.pol_tick = self.robot.lowstate.tick
        if self.run_policy:
            # Inference policy when scale set to non-zero
            if self.action_scale > 0.0:
                self.curr_obs = self._get_observation()
                self.actions = self.policy(obs=self.curr_obs)
            else:
                self.actions = np.zeros(self.robot.num_joints)

            # Update the joint position target
            joint_pos_target = self.actions * self.action_scale + self._joint_offsets

            self.robot.set_target_joint_positions(joint_pos_target)
            # Do logging
            if not self.no_log:
                self.log_dict["low_freq"]["observation"][self.curr_log_ind, :] = self.curr_obs
                self.log_dict["low_freq"]["action"][self.curr_log_ind, :] = self.actions
                self.log_dict["low_freq"]["commands"][self.curr_log_ind, :] = self.commands
                self.log_dict["low_freq"]["tick"][self.curr_log_ind] = self.robot.lowstate.tick
                self.log_dict["low_freq"]["timestamp"][self.curr_log_ind] = time.perf_counter()
                self.curr_log_ind += 1
                if self.curr_log_ind == self.max_log_size:
                    self._save_log()
                    self.curr_lowstate_log_ind = 0
                    self.curr_log_ind = 0

    def _log_lowstate(self):
        self.log_dict["high_freq"]["joint_position"][self.curr_lowstate_log_ind, :] = self.robot.joint_pos
        self.log_dict["high_freq"]["joint_velocity"][self.curr_lowstate_log_ind, :] = self.robot.joint_vel
        self.log_dict["high_freq"]["joint_torque"][self.curr_lowstate_log_ind, :] = [
            self.robot.lowstate.motor_state[i].tau_est for i in range(self.robot.num_joints)
        ]
        self.log_dict["high_freq"]["imu_quaternion"][
            self.curr_lowstate_log_ind, :
        ] = self.robot.lowstate.imu_state.quaternion
        self.log_dict["high_freq"]["imu_gyroscope"][
            self.curr_lowstate_log_ind, :
        ] = self.robot.lowstate.imu_state.gyroscope
        self.log_dict["high_freq"]["imu_accelerometer"][
            self.curr_lowstate_log_ind, :
        ] = self.robot.lowstate.imu_state.accelerometer
        self.log_dict["high_freq"]["foot_force"][self.curr_lowstate_log_ind, :] = self.robot.lowstate.foot_force
        self.log_dict["high_freq"]["tick"][self.curr_lowstate_log_ind] = self.robot.lowstate.tick
        self.log_dict["high_freq"]["timestamp"][self.curr_lowstate_log_ind] = time.perf_counter()

        self.curr_lowstate_log_ind += 1
        if self.curr_lowstate_log_ind == self.max_log_size:
            self._save_log()
            self.curr_lowstate_log_ind = 0
            self.curr_log_ind = 0

    def _save_log(self):
        os.makedirs(self.log_dir, exist_ok=True)
        save_dict = {}
        low_freq_dict = {}
        high_freq_dict = {}
        for key, val in self.log_dict["low_freq"].items():
            if key in ["obs_names", "action_names"]:
                low_freq_dict[key] = val
            else:
                low_freq_dict[key] = val[: self.curr_log_ind, ...]
        for key, val in self.log_dict["high_freq"].items():
            if key in ["joint_order"]:
                high_freq_dict[key] = val
            else:
                high_freq_dict[key] = val[: self.curr_lowstate_log_ind, ...]
        save_dict = {"low_freq": low_freq_dict, "high_freq": high_freq_dict, "policy": self.log_dict["policy"]}
        print()
        print(f"Saving log data to {self.log_dir}/log_{self.save_ind}.pkl")
        with open(self.log_dir + f"/log_{self.save_ind}.pkl", "wb") as handle:
            pickle.dump(save_dict, handle)
        self.save_ind += 1

    def handle_controller_input(self):
        # Release the robot and start the policy
        if self.robot.controller.Start:
            print("Overriding factory controller")
            self.robot.override_factory_controller()
            self.run_policy = True

        # Stop the policy and switch to normal mode
        elif self.robot.controller.A:
            print("Activating factory controller")
            self.run_policy = False
            self.robot.activate_factory_controller()

        # Dampen the robot
        elif self.robot.controller.B:
            print("Damping")
            self.run_policy = False
            self.robot.dampen_robot()

    def handle_keyboard_interrupt(self):
        print("Exiting...")
        self.run_policy = False
        if not self.no_log:
            self._save_log()
        self.robot.activate_factory_controller()
        time.sleep(1)
        sys.exit(-1)

    def run_main_loop(self):
        pol_time = time.perf_counter()
        current_tick = 0
        try:
            while True:
                start_t = time.perf_counter()
                self.handle_controller_input()
                # Run policy at control_dt rate
                update_time = time.perf_counter() - pol_time
                if update_time >= self.control_dt - self.dt_threshold:
                    prev_pol_tick = self.pol_tick
                    self._set_target_joint_positions()
                    pol_time = time.perf_counter()
                    print(
                        f"Commands: {self.commands}, delay: {update_time - self.control_dt: .03e}, "
                        f"lowstate delay: {(self.pol_tick - prev_pol_tick) / 1000 - self.control_dt: .01e}",
                        end="\r",
                    )
                # Only log if policy is running. If get new state, then log
                if not self.no_log and self.run_policy and self.robot.lowstate.tick != current_tick:
                    self._log_lowstate()
                current_tick = self.robot.lowstate.tick
                # Run main thread at 1000Hz
                delaytime = 1 / 1000 - (time.perf_counter() - start_t)
                time.sleep(max(0, delaytime))
        except KeyboardInterrupt:
            self.handle_keyboard_interrupt()
