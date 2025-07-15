# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Agility robots.

The following configurations are available:

* :obj:`DIGIT_CFG`: Agility Digit robot with simple PD controller for the legs and arms

"""
import os

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg

import drail_extensions

from .default_joint_params import default_joint_pos

# from isaaclab.actuators import ImplicitActuatorCfg, DelayedPDActuatorCfg
from .delayed_actuator import DelayedPDActuatorCfg

USD_ASSETS_DIR = os.path.join(os.Path(drail_extensions.__file__).parent, "core/data/assets/robots")

DIGIT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{USD_ASSETS_DIR}/digit/digit.usda",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=None,
            max_angular_velocity=None,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=1
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.01),
        joint_pos=default_joint_pos,
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=1.0,
    actuators={
        "all": DelayedPDActuatorCfg(
            min_delay=4,
            max_delay=8,
            joint_names_expr=[".*leg_hip.*", ".*leg_knee.*", ".*leg_toe_a_joint.*", ".*leg_toe_b_joint.*", ".*arm.*"],
            effort_limit={
                ".*leg_hip_roll.*": 126.682458,
                ".*leg_hip_yaw.*": 79.176536,
                ".*leg_hip_pitch.*": 216.927898,
                ".*leg_knee.*": 231.31695,
                ".*leg_toe_a_joint.*": 41.975942,
                ".*leg_toe_b_joint.*": 41.975942,
                ".*arm_shoulder_roll.*": 126.682458,
                ".*arm_shoulder_pitch.*": 126.682458,
                ".*arm_shoulder_yaw.*": 79.176536,
                ".*arm_elbow.*": 126.682458,
            },
            stiffness={
                ".*leg_hip_roll.*": 80.0,
                ".*leg_hip_yaw.*": 80.0,
                ".*leg_hip_pitch.*": 110.0,
                ".*leg_knee.*": 140.0,
                ".*leg_toe_a_joint.*": 40.0,
                ".*leg_toe_b_joint.*": 40.0,
                ".*arm_shoulder_roll.*": 80.0,
                ".*arm_shoulder_pitch.*": 80.0,
                ".*arm_shoulder_yaw.*": 50.0,
                ".*arm_elbow.*": 80.0,
            },
            damping={
                ".*leg_hip_roll.*": 8.0,
                ".*leg_hip_yaw.*": 8.0,
                ".*leg_hip_pitch.*": 10.0,
                ".*leg_knee.*": 12.0,
                ".*leg_toe_a_joint.*": 6.0,
                ".*leg_toe_b_joint.*": 6.0,
                ".*arm_shoulder_roll.*": 9.0,
                ".*arm_shoulder_pitch.*": 9.0,
                ".*arm_shoulder_yaw.*": 7.0,
                ".*arm_elbow.*": 9.0,
            },
            armature={
                ".*leg_hip_roll.*": 0.173824,
                ".*leg_hip_yaw.*": 0.0679,
                ".*leg_hip_pitch.*": 0.120473,
                ".*leg_knee.*": 0.120473,
                ".*leg_toe_a_joint.*": 0.036089,
                ".*leg_toe_b_joint.*": 0.036089,
            },
        ),
    },
)
