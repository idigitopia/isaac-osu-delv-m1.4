# import warnings

import numpy as np
from remote_controller import RemoteController
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
    MotionSwitcherClient,
)
from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.idl.default import (
    unitree_go_msg_dds__LowCmd_,
    unitree_go_msg_dds__LowState_,
)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC


class Robot:
    def __init__(self, network):
        if network is not None:
            ChannelFactoryInitialize(0, network)
        else:
            ChannelFactoryInitialize(0)

        # Initialize the controller
        self.controller = RemoteController()

        # Create message containers
        self.lowcmd = unitree_go_msg_dds__LowCmd_()
        self.lowstate = unitree_go_msg_dds__LowState_()
        self.update_t = 0

        self.num_joints = 12

        # PD gains
        self._KP = np.array([25] * self.num_joints)
        self._KD = np.array([0.5] * self.num_joints)

        # Gravity vector
        self.GRAVITY_VEC = np.array([0, 0, -1.0])

        # Low level command constants
        self.HIGHLEVEL = 0xEE
        self.LOWLEVEL = 0xFF
        self.TRIGERLEVEL = 0xF0
        self.PosStopF = 2.146e9
        self.VelStopF = 16000.0

        self.crc = CRC()

        self._init()

    def _init(self):
        self._init_lowcmd()
        self._init_pubsub()
        self._init_sport_client()
        self._init_motion_switcher_client()
        self.msc.SelectMode("normal")

    def _init_pubsub(self):

        # Create a publisher to send commands to the robot
        self.cmd_pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.cmd_pub.Init()

        # Create a subscriber to receive state from the robot
        self.state_sub = ChannelSubscriber("rt/lowstate", LowState_)
        # Set queue length to be 0, so we always get the most update to date state
        self.state_sub.Init(self._state_callback, 0)

    def _init_sport_client(self):
        # Sport client
        self.sc = SportClient()
        self.sc.SetTimeout(5.0)
        self.sc.Init()

    def _init_motion_switcher_client(self):
        # Motion switcher client
        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()

    def _init_lowcmd(self):
        self.lowcmd.head[0] = 0xFE
        self.lowcmd.head[1] = 0xEF
        self.lowcmd.level_flag = 0xFF
        self.lowcmd.gpio = 0
        for i in range(20):
            self.lowcmd.motor_cmd[i].mode = 0x01  # (PMSM) mode
            self.lowcmd.motor_cmd[i].q = self.PosStopF
            self.lowcmd.motor_cmd[i].dq = self.VelStopF
            self.lowcmd.motor_cmd[i].kp = 0
            self.lowcmd.motor_cmd[i].kd = 0
            self.lowcmd.motor_cmd[i].tau = 0

    def _reset_lowcmd(self):
        for i in range(self.num_joints):
            self.lowcmd.motor_cmd[i].kp = self._KP[i]
            self.lowcmd.motor_cmd[i].kd = self._KD[i]
            self.lowcmd.motor_cmd[i].dq = 0
            self.lowcmd.motor_cmd[i].tau = 0

    def _state_callback(self, msg: LowState_):
        # Update the state of the robot
        self.lowstate = msg
        # Update the remote state
        self.controller.parse(self.lowstate.wireless_remote)

    def activate_factory_controller(self):
        self.msc.SelectMode("normal")
        self._reset_lowcmd()

    def override_factory_controller(self):
        status, result = self.msc.CheckMode()
        while result["name"]:
            self.msc.ReleaseMode()
            status, result = self.msc.CheckMode()
        self._reset_lowcmd()

    def dampen_robot(self):
        self.sc.Damp()
        self._reset_lowcmd()

    def set_target_joint_positions(self, joint_pos_target):
        for i in range(self.num_joints):
            self.lowcmd.motor_cmd[i].q = joint_pos_target[i]

        # Check CRC
        self.lowcmd.crc = self.crc.Crc(self.lowcmd)

        # Publish the command
        self.cmd_pub.Write(self.lowcmd)

    # States
    @property
    def angular_velocity(self):
        return self.lowstate.imu_state.gyroscope

    @property
    def quaternion(self):
        return self.lowstate.imu_state.quaternion

    @property
    def projected_gravity(self):
        quat = self.quaternion
        q_w = quat[0]
        q_vec = np.array(quat[1:])
        a = self.GRAVITY_VEC * (2.0 * q_w**2 - 1.0)
        b = np.cross(q_vec, self.GRAVITY_VEC) * q_w * 2.0
        c = q_vec * np.dot(q_vec, self.GRAVITY_VEC) * 2.0
        return a - b + c

    @property
    def joint_pos(self):
        return [self.lowstate.motor_state[i].q for i in range(self.num_joints)]

    @property
    def joint_vel(self):
        return [self.lowstate.motor_state[i].dq for i in range(self.num_joints)]
