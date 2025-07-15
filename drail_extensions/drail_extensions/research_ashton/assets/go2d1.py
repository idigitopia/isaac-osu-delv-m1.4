# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

USD_DIR_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "usds")


"""Configuration of Unitree Go2+D1 using DC-Motor actuator model for robot and Implicit for arm."""
GO2D1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{USD_DIR_PATH}/go2d1/go2d1.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.34),
        joint_pos={
            ".*L_hip_joint": 0.1,
            ".*R_hip_joint": -0.1,
            ".*F[L,R]_thigh_joint": 0.8,
            ".*R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
            ".*arm_.*_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_legs": DCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=23.5,
            effort_limit_sim=23.5,
            saturation_effort=23.5,
            velocity_limit=30.0,
            velocity_limit_sim=30.0,
            stiffness=25.0,
            damping=0.5,
            friction=0.0,
        ),
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[".*arm_.*_joint"],
            velocity_limit=100.0,
            velocity_limit_sim=100.0,
            effort_limit=87.0,
            effort_limit_sim=87.0,
            stiffness=800.0,
            damping=40.0,
        ),
    },
)
