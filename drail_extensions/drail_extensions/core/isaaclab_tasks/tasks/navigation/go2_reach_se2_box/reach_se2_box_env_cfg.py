import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import (
    EventTermCfg as EventTerm,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
)
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import FrameTransformerCfg, OffsetCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

import drail_extensions.core.isaaclab_tasks.mdp as mdp
import drail_extensions.core.isaaclab_tasks.tasks.SaW.go2.spot_reward_env_cfg as spot_reward_env_cfg
from drail_extensions.core.isaaclab_tasks.tasks.navigation.go2_reach_se2_command import (
    reach_se2_command_env_cfg,
)

##
# Scene definition
##


@configclass
class MySceneCfg(reach_se2_command_env_cfg.MySceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=400.0),
            scale=(5.0, 5.0, 5.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    base_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        debug_vis=False,
        visualizer_cfg=FRAME_MARKER_CFG.replace(prim_path="/Visuals/EndEffectorFrameTransformer"),
        target_frames=[
            # box operation frame
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/object",
                name="object_operation_frame",
                offset=OffsetCfg(
                    pos=(0.8, 0.0, 0.0),
                    rot=(0.0, 0.0, 0.0, 1.0),
                ),
            ),
        ],
    )

    def __post_init__(self):
        super().__post_init__()
        self.base_frame.visualizer_cfg.markers["frame"].scale = (0.10, 0.10, 0.10)


##
# MDP settings
##


@configclass
class ObservationsCfg(reach_se2_command_env_cfg.ObservationsCfg):
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(reach_se2_command_env_cfg.ObservationsCfg.PolicyCfg):
        """Observations for policy group."""

        box_pos_rel = ObsTerm(
            func=mdp.pos_rel,
            params={
                "frame_cfg": SceneEntityCfg("base_frame"),
                "target_frame_names": ["object_operation_frame"],
                "axes": "xyz",
            },
        )
        box_heading_rel = ObsTerm(
            func=mdp.heading_rel,
            params={
                "frame_cfg": SceneEntityCfg("base_frame"),
                "target_frame_names": ["object_operation_frame"],
                "euler": True,
            },
        )

        def __post_init__(self):
            super().__post_init__()
            self.pose_command = None

    @configclass
    class CriticCfg(reach_se2_command_env_cfg.ObservationsCfg.CriticCfg):
        """Observations for critic group."""

        box_pos_rel = ObsTerm(
            func=mdp.pos_rel,
            params={
                "frame_cfg": SceneEntityCfg("base_frame"),
                "target_frame_names": ["object_operation_frame"],
                "axes": "xyz",
            },
        )
        box_heading_rel = ObsTerm(
            func=mdp.heading_rel,
            params={
                "frame_cfg": SceneEntityCfg("base_frame"),
                "target_frame_names": ["object_operation_frame"],
                "euler": True,
            },
        )

        def __post_init__(self):
            super().__post_init__()
            self.pose_command = None

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventsCfg(spot_reward_env_cfg.EventCfg):
    """Configuration for events."""

    reset_object = EventTerm(
        func=mdp.reset_root_state_uniform_restricted,
        mode="reset",
        params={
            "pose_range": {"x": [-3.0, 3.0], "y": [-3.0, 3.0], "z": [-0.0, 0.0], "yaw": [-3.14, 3.14]},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names=".*"),
            "asset_cfg_restricted": SceneEntityCfg("robot"),
            "min_distance": 0.8,
        },
    )


@configclass
class RewardsCfg:
    # Reward for arm base (with robot) to move to operation space of box
    position_tracking = RewTerm(
        func=mdp.pos_rel_reward_tanh,
        weight=0.5,
        params={
            "std": 2.0,
            "frame_cfg": SceneEntityCfg("base_frame"),
            "target_frame_name": "object_operation_frame",
            "axes": "xy",
        },
    )
    position_tracking_fine_grained = RewTerm(
        func=mdp.pos_rel_reward_tanh,
        weight=0.5,
        params={
            "std": 0.2,
            "frame_cfg": SceneEntityCfg("base_frame"),
            "target_frame_name": "object_operation_frame",
            "axes": "xy",
        },
    )
    heading_tracking = RewTerm(
        func=mdp.heading_rel_reward_tanh,
        weight=0.5,
        params={
            "std": 2.0,
            "frame_cfg": SceneEntityCfg("base_frame"),
            "target_frame_name": "object_operation_frame",
        },
    )


# Environment configuration
##


@configclass
class Go2_Reach_SE2_Box_EnvCfg(reach_se2_command_env_cfg.Go2_Reach_SE2_Command_EnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    events: EventsCfg = EventsCfg()

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()
        # general settings
        self.episode_length_s = 8
        self.commands = None


@configclass
class Go2_Reach_SE2_Box_EnvCfg_PLAY(Go2_Reach_SE2_Box_EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4
        self.scene.env_spacing = 4.0

        self.scene.contact_forces.debug_vis = True
        self.scene.terrain.debug_vis = True
        self.scene.base_frame.debug_vis = True
        self.sim.render_interval = reach_se2_command_env_cfg.Go2_Reach_SE2_Command_EnvCfg_PLAY().sim.render_interval

        self.events.move_object = EventTerm(
            mode="interval",
            interval_range_s=(0.0, 0.0),
            func=mdp.move_asset_by_gui_keyboard,
            params={
                "asset_cfg": SceneEntityCfg("object"),
                "offset": (0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
                "pos_sensitivity": 0.1,
                "rot_sensitivity": 0.1,
            },
        )
