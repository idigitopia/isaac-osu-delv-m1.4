# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import (
    ArticulationCfg,
    AssetBaseCfg,
    RigidObjectCfg,
    RigidObjectCollectionCfg,
)
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import (
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, CameraCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import drail_extensions.research_ashton.mdp as mdp
from drail_extensions.research_ashton.assets import *

##
# Scene definition
##


JOINT_ORDER = ["FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"]


# Calculate positions in a circle
OBJECT_SPAWN_RADIUS = 4.0  # radius of the circle
OBJECT_SPAWN_NUM = 8  # number of fixed objects (excluding random)
OBJECT_SPAWN_POSITIONS = [
    (OBJECT_SPAWN_RADIUS * math.cos(2 * math.pi * i / OBJECT_SPAWN_NUM),
        OBJECT_SPAWN_RADIUS * math.sin(2 * math.pi * i / OBJECT_SPAWN_NUM),
        1.0) for i in range(OBJECT_SPAWN_NUM)
]


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    #########################################################
    # ground terrain
    # This is for the Small Warehouse Environment
    #########################################################
    # ground = AssetBaseCfg(
    #     prim_path="/World/Ground",
    #     spawn=sim_utils.UsdFileCfg(
    #         # usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Terrains/flat_plane.usd",
    #         usd_path= "/home/drail/Desktop/empty_warehouse_v2.usd",
    #         scale=(1, 1, 1),
    #     ),
    #     init_state=ArticulationCfg.InitialStateCfg(
    #         pos=(0.0, 0.0, 0.001),
    #     ),
    # )
    #########################################################


    
    #########################################################
    # ground terrain
    # This is for the Graces Quarters Environment
    #########################################################
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="usd",
        # terrain_generator=ROUGH_TERRAINS_CFG,
        usd_path=f"/home/drail/Desktop/gq_data/Worlds/GQ_recentered2.usd",
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=(
                f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/"
                "TilesMarbleSpiderWhiteBrickBondHoned.mdl"
            ),
            project_uvw=True,
            texture_scale=(1, 1),
        ),
        debug_vis=False,
    )
    #########################################################




    #########################################################
    # ground terrain
    # This is for the Flat Plane Environment
    #########################################################
    # # ground terrain
    # terrain = TerrainImporterCfg(
    #     prim_path="/World/ground",
    #     terrain_type="plane",
    #     terrain_generator=None,
    #     max_init_terrain_level=5,
    #     collision_group=-1,
    #     physics_material=sim_utils.RigidBodyMaterialCfg(
    #         friction_combine_mode="multiply",
    #         restitution_combine_mode="multiply",
    #         static_friction=1.0,
    #         dynamic_friction=1.0,
    #     ),
    #     visual_material=sim_utils.MdlFileCfg(
    #         mdl_path=(
    #             f"{ISAAC_NUCLEUS_DIR}/Materials/Base/Concrete/ConcretePolished"
    #             "/ConcretePolished.mdl"
    #         ),
    #         project_uvw=True,
    #         texture_scale=(2.0, 2.0),  # Larger scale for better detail visibility
    #     ),
    #     debug_vis=False,
    # )
    # robots
    robot: ArticulationCfg = GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # # sensors
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        debug_vis=True,
        )

    rgbd_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Head_upper/Camera",
        offset=CameraCfg.OffsetCfg(pos=(0.1, 0.0, 0.1), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
        data_types=["rgb", "depth"],
        depth_clipping_behavior="max",
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=1.88,
            focus_distance=0.5,
            horizontal_aperture=3.896,
            vertical_aperture=2.453,
            clipping_range=(0.01, 100.0),
        ),
        width=1280,
        height=720,
    )

    # lights
    # sky_light = AssetBaseCfg(
    #     prim_path="/World/skyLight",
    #     spawn=sim_utils.DomeLightCfg(
    #         intensity=5000.0,
    #         texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
    #     ),
    # )

    # object collection
    objects: RigidObjectCollectionCfg = RigidObjectCollectionCfg(
        rigid_objects={
            "small_box": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Small_Box",
                spawn=sim_utils.CuboidCfg(
                    size=(0.3, 0.3, 0.3),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.2), metallic=0.2),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=4, solver_velocity_iteration_count=0
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=OBJECT_SPAWN_POSITIONS[0]),
            ),
            "big_box": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Big_Box",
                spawn=sim_utils.CuboidCfg(
                    size=(0.5, 0.5, 0.5),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0), metallic=0.2),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=4, solver_velocity_iteration_count=0
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=OBJECT_SPAWN_POSITIONS[1]),
            ),
            
            "small_sphere": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Small_Sphere",
                spawn=sim_utils.SphereCfg(
                    radius=0.2,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=4, solver_velocity_iteration_count=0
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=OBJECT_SPAWN_POSITIONS[2]),
            ),
            "big_sphere": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Big_Sphere",
                spawn=sim_utils.SphereCfg(
                    radius=0.5,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0), metallic=0.2),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=4, solver_velocity_iteration_count=0
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=OBJECT_SPAWN_POSITIONS[3]),
            ),
            "small_cone": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Small_Cone",
                spawn=sim_utils.ConeCfg(
                    radius=0.2,
                    height=0.3,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=4, solver_velocity_iteration_count=0
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=OBJECT_SPAWN_POSITIONS[4]),
            ),
            "big_cone": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Big_Cone",
                spawn=sim_utils.ConeCfg(
                    radius=0.4,
                    height=0.6,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 1.0), metallic=0.2),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=4, solver_velocity_iteration_count=0
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=OBJECT_SPAWN_POSITIONS[5]),
            ),
            "small_cylinder": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Small_Cylinder",
                spawn=sim_utils.CylinderCfg(
                    radius=0.1,
                    height=0.3,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0), metallic=0.2),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=4, solver_velocity_iteration_count=0
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=OBJECT_SPAWN_POSITIONS[6]),
            ),
            "big_cylinder": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Big_Cylinder",
                spawn=sim_utils.CylinderCfg(
                    radius=0.4,
                    height=0.6,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0), metallic=0.2),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=4, solver_velocity_iteration_count=0
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=OBJECT_SPAWN_POSITIONS[7]),
            ),
            "random": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Random_Object",
                spawn=sim_utils.MultiAssetSpawnerCfg(
                assets_cfg=[
                    sim_utils.ConeCfg(
                        radius=0.3,
                        height=0.8,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 1.0, 0.5), metallic=0.2),
                    ),
                    sim_utils.CuboidCfg(
                        size=(0.5, 0.3, 0.3),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 1.0, 0.5), metallic=0.2),
                    ),
                    sim_utils.SphereCfg(
                        radius=0.3,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 1.0, 0.5), metallic=0.2),
                    ),
                    sim_utils.CylinderCfg(
                        radius=0.2,
                        height=0.8,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 1.0, 0.5), metallic=0.2),
                    ),
                ],
                random_choice=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=0, solver_velocity_iteration_count=0
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos= (OBJECT_SPAWN_POSITIONS[7][0] + 2.0, OBJECT_SPAWN_POSITIONS[7][1], OBJECT_SPAWN_POSITIONS[7][2]  + 2.0)),  # Random object in the center
        )
        }
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    ## Just matters we have a command description of (v_x, v_y, omega_z, stand_bit)
    ## Commands should be overridden in the play wrapper
    base_velocity = mdp.CategoricalCommandCfg(
        asset_name="robot",
        body_name="base",
        resampling_time_range=(5.0, 10.0),
        debug_vis=True,
        just_vel_cmd=True,
        ranges={
            "stand": mdp.CategoricalCommandCfg.Ranges(
                lin_vel_x=(-0.0, 0.0),
                lin_vel_y=(-0.0, 0.0),
                ang_vel_z=(-0.0, 0.0),
                height=(0.35, 0.35),
                roll=(-0.0, 0.0),
                pitch=(-0.0, 0.0),
                probability=1.0,
            )
        },
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=JOINT_ORDER, scale=0.25, use_default_offset=True, preserve_order=True
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            params={"asset_cfg": SceneEntityCfg("robot")}
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_ORDER, preserve_order=True)},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_ORDER, preserve_order=True)},
        )
        actions = ObsTerm(
            func=mdp.last_action,
            params={"action_name": "joint_pos"}
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"}
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            noise=Unoise(n_min=-0.1, n_max=0.1),
            params={"asset_cfg": SceneEntityCfg("robot")}
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            params={"asset_cfg": SceneEntityCfg("robot")}
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_ORDER, preserve_order=True)},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_ORDER, preserve_order=True)},
        )
        actions = ObsTerm(
            func=mdp.last_action,
            params={"action_name": "joint_pos"}
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"}
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True


    @configclass
    class CameraImageCfg(ObsGroup):
        """Observations for camera image group."""

        rgb_image = ObsTerm(
            func=mdp.image_with_vis,
            params={
                "sensor_cfg": SceneEntityCfg("rgbd_camera"),
                "data_type": "rgb",
                "channel_first": True,
                "debug_vis": True,
                "normalize": False,
            },
        )
        depth_image = ObsTerm(
            func=mdp.image_with_vis,
            params={
                "sensor_cfg": SceneEntityCfg("rgbd_camera"),
                "data_type": "depth",
                "channel_first": True,
                "debug_vis": True,
                "normalize": True,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    camera: CameraImageCfg = CameraImageCfg()


@configclass
class EventCfg:
    """Configuration for events during evaluation/play without DR."""

    # Root pose
    reset_base = EventTerm(
        func = mdp.reset_root_state_uniform,
        mode = "reset",
        # Need to reset root pose everytime
        min_step_count_between_reset = 0,
        params = {
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    # Object pose
    reset_objects = EventTerm(
        func=mdp.reset_object_pose_uniform,
        mode="reset",
        min_step_count_between_reset=0,
        params={
            "pose_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "yaw": (-math.pi, math.pi)},
            "asset_cfg": SceneEntityCfg("objects")
        },
    )

    # Joint positions
    reset_robot_joints = EventTerm(
        func = mdp.reset_joints_by_scale,
        mode = "reset",
        # Need to reset joint positions everytime
        min_step_count_between_reset = 0,
        params = {
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp_spot,
        weight=5.0,
        params={"command_name": "base_velocity",
                "std": math.sqrt(1.0), "ramp_rate": 0.5, "ramp_at_vel": 1.0,
                "asset_cfg": SceneEntityCfg("robot")},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp_spot,
        weight=5.0,
        params={"command_name": "base_velocity", "std": math.sqrt(2.0), "asset_cfg": SceneEntityCfg("robot")},
    )

    # -- penalties
    action_smoothness = RewTerm(func=mdp.action_smoothness_penalty, weight=-1.0)
    base_motion = RewTerm(
        func=mdp.base_motion_penalty,
        weight=-4.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "lin_weight": 0.5}
    )
    base_orientation = RewTerm(
        func=mdp.base_orientation_penalty,
        weight=-3.0, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    joint_torques = RewTerm(
        func=mdp.joint_torques_penalty,
        weight=-5.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    joint_pos = RewTerm(
        func=mdp.joint_position_penalty,
        weight=-0.7,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_thigh_joint", ".*_calf_joint"]),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.5,
            "command_name": "base_velocity",
        },
    )
    hip_pos = RewTerm(
        func=mdp.joint_position_penalty,
        weight=-1.4,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint"]),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.5,
            "command_name": "base_velocity",
        },
    )
    joint_acc = RewTerm(
        func=mdp.joint_acceleration_penalty,
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    joint_vel = RewTerm(
        func=mdp.joint_velocity_penalty,
        weight=-1.0e-2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )

    # -- gait related
    air_time_variance = RewTerm(
        func=mdp.air_time_variance_penalty,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot"),
            "command_name": "base_velocity"},
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_v2,
        weight=5.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot"),
            "threshold": 1.0,
        },
    )
    feet_clearance = RewTerm(
        func=mdp.foot_clearance_reward,
        weight=0.5,
        params={
            "command_name": "base_velocity",
            "std": math.sqrt(0.05),
            "tanh_mult": 2.0,
            "target_height": 0.06,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*foot"),
        },
    )
    gait = RewTerm(
        func=mdp.GaitReward,
        weight=5.0,
        params={
            "command_name": "base_velocity",
            "std": math.sqrt(0.1),
            "max_err": 0.2,
            "synced_feet_pair_names": (("FL_foot", "RR_foot"), ("FR_foot", "RL_foot")),
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces"),
        },
    )
    feet_contact = RewTerm(
        func=mdp.feet_contact_quadruped,
        weight=5.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot"),
        },
    )
    feet_slip = RewTerm(
        func=mdp.foot_slip_penalty,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot"),
            "threshold": 1.0,
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*base", ".*Head_lower"]),
                "threshold": 1.0},
    )
    # base_height = DoneTerm(
    #     func=mdp.base_height_out_of_range,
    #     params={"asset_cfg": SceneEntityCfg("robot"), "min_height": 0.3}
    # )


# Environment configuration
##


@configclass
class Go2PoseWrapperEnvCfg_PLAY(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=10.0, replicate_physics=False)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4 # 50 Hz
        self.episode_length_s = 5000.0
        self.scene.num_envs = 1
        # simulation settings
        self.sim.dt = 1/200 # 200 Hz
        self.sim.render_interval = self.decimation * 2
        self.sim.disable_contact_processing = True
        # self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        if self.scene.rgbd_camera is not None:
            self.scene.rgbd_camera.update_period = self.sim.dt * self.decimation
