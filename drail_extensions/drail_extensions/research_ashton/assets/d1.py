# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from .delayed_implicit import DelayedImplicitActuatorCfg

USD_DIR_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "usds")


""" D1 base configuration using TBD actuator model."""
D1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # usd_path=f"{USD_DIR_PATH}/d1/d1.usd",
        usd_path=f"{USD_DIR_PATH}/d1_550/d1_550_description.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=1.0,
    # actuators={
    #     "arm": DelayedImplicitActuatorCfg(
    #         min_delay=4,
    #         max_delay=8,
    #         joint_names_expr=[".*"],
    #         velocity_limit=5.0,
    #         velocity_limit_sim=5.0,
    #         effort_limit={
    #             "arm_[1-2]_joint": 3.3,
    #             "arm_[3-6]_joint": 1.7,
    #             "arm_7_[1-2]_joint": 1.7,
    #         },
    #         effort_limit_sim={
    #             "arm_[1-2]_joint": 3.3,
    #             "arm_[3-6]_joint": 1.7,
    #             "arm_7_[1-2]_joint": 1.7,
    #         },
    #         stiffness=50.0,
    #         damping=5.0,
    #     ),
    # },
    actuators={
        # "arm": ImplicitActuatorCfg(
        #     joint_names_expr=[".*"],
        #     velocity_limit=5.0,
        #     velocity_limit_sim=5.0,
        #     effort_limit={
        #         "arm_[1-2]_joint": 3.3,
        #         "arm_[3-6]_joint": 1.7,
        #         "arm_7_[1-2]_joint": 1.7,
        #     },
        #     effort_limit_sim={
        #         "arm_[1-2]_joint": 3.3,
        #         "arm_[3-6]_joint": 1.7,
        #         "arm_7_[1-2]_joint": 1.7,
        #     },
        #     stiffness=50.0,
        #     damping=5.0,
        # ),
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            velocity_limit=5.0,
            velocity_limit_sim=5.0,
            effort_limit={
                "Joint[1-2]": 3.3,
                "Joint[3-6]": 1.7,
                "Joint_[L-R]": 1.7,
            },
            effort_limit_sim={
                "Joint[1-2]": 3.3,
                "Joint[3-6]": 1.7,
                "Joint_[L-R]": 1.7,
            },
            stiffness=50.0,
            damping=5.0,
        ),
    },
)
