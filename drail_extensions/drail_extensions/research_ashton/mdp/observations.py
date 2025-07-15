# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create observation terms.

The functions can be passed to the :class:`isaaclab.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Sequence

import cv2
import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import numpy as np
import rerun
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import POSITION_GOAL_MARKER_CFG
from isaaclab.sensors import RayCasterCamera, TiledCamera
from isaaclab.utils.noise import NoiseCfg

from drail_extensions.research_ashton.utils import mesh_utils, visualization_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def relative_poses(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    asset_cfg_other: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Return the relative pose of asset_cfg_other with respect to asset_cfg."""
    asset: Articulation = env.scene[asset_cfg.name]
    asset_other: Articulation = env.scene[asset_cfg_other.name]

    target_vec = asset_other.data.root_pos_w[:, :3] - asset.data.root_pos_w[:, :3]
    pos_command_b = math_utils.quat_rotate_inverse(math_utils.yaw_quat(asset.data.root_quat_w), target_vec)
    heading_command_b = math_utils.wrap_to_pi(asset_other.data.heading_w - asset.data.heading_w)

    return torch.cat([pos_command_b, heading_command_b.unsqueeze(1)], dim=-1)


class point_cloud_in_mesh_frame(ManagerTermBase):
    """Computes the point cloud of a mesh in its local frame.

    This class accepts a mesh primitive path, which can be specified as a regex pattern, to identify and compute
    the point cloud for all matching mesh primitives. The result is returned as a batch of point clouds with the
    shape (number of matching mesh primitives, number of points, 3). For example, to compute the point cloud of a
    mesh in each environment, you can use a regex path like "/World/envs/env_*/object/visuals/mesh", which would
    return a point cloud of shape (num_envs, num_points, 3).
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # Create a group of meshes for each mesh prim path in regex format
        mesh_path = self.cfg.params["mesh_prim_path"].format(ENV_REGEX_NS=self._env.scene.env_regex_ns)
        self._matching_prims = sim_utils.find_matching_prims(mesh_path)
        if not self._matching_prims:
            raise ValueError(f"No matching prims found for mesh path: {mesh_path}")

        # Create a buffer to store point clouds for each mesh group.
        # Shape is (num_envs, num_mesh_groups (size of mesh_prim_paths), num_points, 3)
        self._point_clouds = torch.zeros(len(self._matching_prims), self.cfg.params["num_points"], 3).to(self.device)

        # Setup visualizers for each mesh group
        if cfg.params["debug_vis"]:
            colors = cfg.params["debug_vis_args"].get("colors", (1.0, 1.0, 1.0))
            radius = cfg.params["debug_vis_args"].get("radius", 0.005)
            self.pc_visualizer = visualization_utils.get_point_cloud_visualizer_cfg(
                color=colors,
                radius=radius,
            )

    def _compute_point_cloud(self, env: ManagerBasedEnv, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = slice(None)
        for i, mesh_prim in enumerate(self._matching_prims):
            self._point_clouds[i] = torch.from_numpy(
                mesh_utils.get_point_cloud_from_mesh(prim=mesh_prim, num_points=self.cfg.params["num_points"])
            )

    def __call__(
        self,
        env: ManagerBasedEnv,
        mesh_prim_path: str = "",
        debug_vis: bool = False,
        debug_vis_args: dict[str, Any] = {"colors": (1.0, 1.0, 1.0), "radius": 0.005},
        num_points: int = 1000,
    ) -> torch.Tensor:
        self._compute_point_cloud(env)

        # Visualize point clouds if debug vis is enabled
        if debug_vis:
            self.pc_visualizer.visualize(self._point_clouds.flatten(0, 1).cpu().numpy())

        return self._point_clouds


class point_cloud_transformed(point_cloud_in_mesh_frame):
    """Computes the point cloud of a mesh in its local frame and then transforms it using the pose of a specified asset.

    This class is useful when you need to compute the point cloud of a mesh associated with a particular asset and
    transform it into the frame where the asset's pose is defined. For instance, if you provide an asset whose pose
    is defined in the world frame along with its mesh primitive path, the resulting point cloud will be computed
    and transformed into the world frame.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # Whether to compute point cloud every call or just at the initialization. For example, for rigid body,
        # we only need to compute once and transform according to its pose every call, however for articulated body,
        # we need to compute every call as its joint configuration changes.
        if not cfg.params.get("compute_every_call", False):
            self._compute_point_cloud(env)

        self._asset: Articulation | RigidObject = env.scene[self.cfg.params["asset_cfg"].name]

    def __call__(
        self,
        env: ManagerBasedEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("asset"),
        mesh_prim_path: str = "",
        debug_vis: bool = False,
        compute_every_call: bool = False,
        debug_vis_args: dict[str, Any] = {"colors": (1.0, 1.0, 1.0), "radius": 0.005},
        num_points: int = 1000,
    ) -> torch.Tensor:
        if compute_every_call:
            self._compute_point_cloud(env)

        # Transform point cloud of object's mesh from its root frame to world frame
        point_cloud = math_utils.quat_rotate(
            self._asset.data.root_quat_w.unsqueeze(1), self._point_clouds
        ) + self._asset.data.root_pos_w.unsqueeze(1)

        # Visualize point clouds if debug vis is enabled
        if debug_vis:
            self.pc_visualizer.visualize(point_cloud.flatten(0, 1).cpu().numpy())

        return point_cloud


class point_cloud_in_reference_frame(point_cloud_transformed):
    """Transforms the point cloud by the relative pose of asset with respect to reference asset.

    This class computes the point cloud of a mesh in its local frame, transforms it by the relative pose of asset with
    respect to reference asset. This results in a point cloud that is observed from the reference asset's frame.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self._reference_asset: Articulation | RigidObject = env.scene[self.cfg.params["reference_asset_cfg"].name]

    def __call__(
        self,
        env: ManagerBasedEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        reference_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        mesh_prim_path: str = "",
        debug_vis: bool = False,
        debug_vis_args: dict[str, Any] = {},
        compute_every_call: bool = False,
        num_points: int = 1000,
        channel_first: bool = True,
    ) -> torch.Tensor:
        if compute_every_call:
            self._compute_point_cloud(env)

        # Compute the relative position of the asset with respect to the reference asset in the world frame
        rel_vec_w = self._asset.data.root_pos_w[:, :3] - self._reference_asset.data.root_pos_w[:, :3]

        # Translate the relative position into the reference asset's frame
        rel_vec_b = math_utils.quat_rotate_inverse(self._reference_asset.data.root_quat_w, rel_vec_w)

        # Compute the relative rotation of the asset with respect to the reference asset in the world frame
        rel_quat_b = math_utils.quat_mul(
            math_utils.quat_inv(self._reference_asset.data.root_quat_w), self._asset.data.root_quat_w
        )

        # Rotate the point cloud into the reference asset's frame
        point_cloud = math_utils.quat_rotate(rel_quat_b.unsqueeze(1), self._point_clouds) + rel_vec_b.unsqueeze(1)

        # Visualize point clouds if debug visualization is enabled
        if debug_vis:
            self.pc_visualizer.visualize(point_cloud.flatten(0, 1).cpu().numpy())

        # Return the point cloud with channels first if specified
        if channel_first:
            return point_cloud.transpose(1, 2).contiguous()
        return point_cloud


def image_with_vis(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
    data_type: str = "rgb",
    noises: list[NoiseCfg] = [],
    convert_perspective_to_orthogonal: bool = False,
    normalize: bool = True,
    channel_first: bool = False,
    debug_vis: bool = False,
    debug_vis_args: dict[str, Any] = {
        "env_ids": slice(None),
        "desired_height": 800,
        "desired_width": 800,
    },
) -> torch.Tensor:
    """Add visualization to the base implementation of image mdp"""

    images = mdp.image(env, sensor_cfg, data_type, convert_perspective_to_orthogonal, normalize)

    for noise in noises or []:
        images = noise.func(images, noise)

    if debug_vis:
        env_ids = debug_vis_args["env_ids"]

        if "distance_to" in data_type or "depth" in data_type:
            for i, depth_map in enumerate(images[env_ids]):
                depth_map = depth_map.squeeze(0).squeeze(-1).cpu().numpy()

                # Normalize depth map to 0-255 range and convert to uint8
                depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
                depth_map = (depth_map * 255).astype(np.uint8)

                # color it
                depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_VIRIDIS)
                depth_map = cv2.resize(
                    depth_map,
                    (debug_vis_args["desired_width"], debug_vis_args["desired_height"]),
                    interpolation=cv2.INTER_LINEAR,
                )

                # Create a resizable window
                window_name = f"Depth Map {i}"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.imshow(window_name, depth_map)

                cv2.waitKey(1)
        elif data_type == "rgb":
            for i, rgb_image in enumerate(images[env_ids]):
                rgb_image = rgb_image.cpu().numpy()
                rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

                rgb_image = cv2.resize(
                    rgb_image,
                    (debug_vis_args["desired_width"], debug_vis_args["desired_height"]),
                    interpolation=cv2.INTER_LINEAR,
                )

                window_name = f"RGB Image {i}"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.imshow(window_name, rgb_image)

                cv2.waitKey(1)
        else:
            print(f"[WARNING] Debug vis not supported for data type: {data_type}")

    if channel_first:
        # Change shape from (B, H, W, C) to (B, C, H, W)
        return images.permute(0, 3, 1, 2).clone()

    return images.clone()


class partial_point_cloud_from_depth_map(ManagerTermBase):
    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        if cfg.params["debug_vis"]:
            if cfg.params["debug_vis_args"].get("native_visualizer", False):
                dot_visualizer_cfg: VisualizationMarkersCfg = POSITION_GOAL_MARKER_CFG.replace(
                    prim_path="/Visuals/PointCloud/points"
                )
                if "color" in cfg.params["debug_vis_args"]:
                    dot_visualizer_cfg.markers["target_far"].visual_material.diffuse_color = cfg.params[
                        "debug_vis_args"
                    ]["color"]
                self.dot_visualizer = VisualizationMarkers(dot_visualizer_cfg)
                self.dot_visualizer.set_visibility(True)
            else:
                rerun.init("partial-pointcloud", spawn=True)

        self._camera: TiledCamera = env.scene.sensors[cfg.params["sensor_cfg"].name]

        self._num_points = cfg.params["num_points"]
        self._max_points = math.prod(self._camera.image_shape)

        # Camera must be in world frame convention. For different types, orientation delta must be changed accordingly
        assert self._camera.cfg.offset.convention == "world", "Camera must be in world frame convention"
        # Z-buffer / Z-axis to forward view / X-axis orientation delta
        self._orientation_delta = math_utils.quat_from_euler_xyz(
            roll=torch.tensor([-math.pi / 2]), pitch=torch.tensor([0.0]), yaw=torch.tensor([-math.pi / 2])
        ).to(self.device)

        self._resample_random_pts_idx()

    def _resample_random_pts_idx(self, env_ids: Sequence[int] | None = None):
        # If semantic tags are given, we can sample points with replacement from semantic mask, otherwise
        # use random indices
        if self.cfg.params.get("semantic_tags_filter") is not None:
            return
        if env_ids is None:
            env_ids = slice(None)
            num_envs = self._env.num_envs
        else:
            num_envs = env_ids.size(0)

        if hasattr(self, "_random_pts_idx"):
            self._random_pts_idx[env_ids] = torch.randint(
                0, self._max_points, (num_envs, self._num_points), device=self.device
            )
        else:
            self._random_pts_idx = torch.randint(0, self._max_points, (num_envs, self._num_points), device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Resample the random points at every reset."""
        self._resample_random_pts_idx(env_ids)

    def __call__(
        self,
        env: ManagerBasedEnv,
        # Depth camera cfg
        sensor_cfg: SceneEntityCfg = SceneEntityCfg("depth_camera"),
        data_type: str = "distance_to_image_plane",
        noises: list[NoiseCfg] = [],
        # Point cloud cfg
        num_points: int = None,
        channel_first: bool = False,
        semantic_tags_filter: list[tuple[str, str]] | None = None,
        # Debug vis cfg
        debug_vis: bool = False,
        debug_vis_args: dict[str, Any] = {
            "env_ids": slice(None),
            "native_visualizer": False,
            "visualize_point_cloud": False,
            "visualize_depth_map": False,
            "depth_map_desired_height": 800,
            "depth_map_desired_width": 800,
            "visualize_semantic_segmentation": False,
            "semantic_segmentation_desired_height": 800,
            "semantic_segmentation_desired_width": 800,
        },
    ) -> torch.Tensor:
        if debug_vis:
            env_ids = debug_vis_args.get("env_ids", slice(None))
            native_visualizer = debug_vis_args.get("native_visualizer", False)
            depth_map_desired_height = debug_vis_args.get("depth_map_desired_height", 800)
            depth_map_desired_width = debug_vis_args.get("depth_map_desired_width", 800)
            visualize_point_cloud = debug_vis_args.get("visualize_point_cloud", False)
            visualize_depth_map = debug_vis_args.get("visualize_depth_map", False)
            visualize_semantic_segmentation = debug_vis_args.get("visualize_semantic_segmentation", False)
            semantic_segmentation_desired_height = debug_vis_args.get("semantic_segmentation_desired_height", 800)
            semantic_segmentation_desired_width = debug_vis_args.get("semantic_segmentation_desired_width", 800)

        """Get partial point cloud from depth map."""
        depth_map = image_with_vis(
            env=env,
            sensor_cfg=sensor_cfg,
            data_type=data_type,
            convert_perspective_to_orthogonal=False,
            normalize=False,
            noises=noises,
            debug_vis=debug_vis and visualize_depth_map,
            debug_vis_args=debug_vis
            and visualize_depth_map
            and {
                "env_ids": env_ids,
                "desired_height": depth_map_desired_height,
                "desired_width": depth_map_desired_width,
            },
        )

        if semantic_tags_filter is None:
            # If depth map contains values greater than far clipping range, then make it invalid
            invalid_depth_mask = depth_map >= self._camera.cfg.spawn.clipping_range[1]
        else:
            # If semantic tags are given, then we can sample points with replacement from semantic mask
            tag_to_idx = {
                (k, v): outer_k
                for outer_k, inner_dict in self._camera.data.info["semantic_segmentation"]["idToLabels"].items()
                for k, v in inner_dict.items()
            }
            semantic_tag_to_idx = [tag_to_idx.get(tuple(tag)) for tag in semantic_tags_filter]
            semantic_tag_to_idx = [int(idx) for idx in semantic_tag_to_idx if idx is not None]
            segmented_images = self._camera.data.output["semantic_segmentation"]

            # If segmented image contains values other than semantic tags, then make is invalid
            invalid_depth_mask = (segmented_images == torch.tensor(semantic_tag_to_idx).to(env.device)).sum(
                -1, keepdim=True
            ) == 0

        # Get point cloud from depth map. All nan values are projected to nan point cloud values
        point_clouds = math_utils.unproject_depth(
            torch.where(invalid_depth_mask, torch.nan, depth_map),
            self._camera.data.intrinsic_matrices,
        )

        # Transform point cloud to forward view
        point_clouds = math_utils.transform_points(point_clouds, quat=self._orientation_delta)

        # Sample valid random points from point cloud
        point_clouds = mesh_utils.sample_valid_points(point_clouds, self._num_points)

        # Visualize point cloud if debug vis is enabled
        if debug_vis and visualize_point_cloud:
            if native_visualizer:
                self.dot_visualizer.visualize(point_clouds[env_ids].flatten(0, 1).cpu().numpy())
            else:
                rerun.log(
                    "pointcloud",
                    rerun.Points3D(
                        point_clouds[env_ids].flatten(0, 1).cpu().numpy(),
                        colors=[255, 0, 255],
                        radii=0.005,
                    ),
                )

        if debug_vis and visualize_semantic_segmentation:
            segmented_images = self._camera.data.output["semantic_segmentation"][env_ids]
            for i in range(segmented_images.size(0)):
                seg_img = segmented_images[i].squeeze(-1).cpu().numpy()
                visualization_utils.show_heatmap(
                    seg_img,
                    semantic_segmentation_desired_width,
                    semantic_segmentation_desired_height,
                    f"Semantic Segmentation {i}",
                    max_val=self._camera.cfg.spawn.clipping_range[1],
                )

        if channel_first:
            return point_clouds.transpose(1, 2).contiguous()
        return point_clouds


class partial_point_cloud_from_ray_hits(ManagerTermBase):
    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        if cfg.params["debug_vis"]:
            if cfg.params["debug_vis_args"].get("native_visualizer", False):
                dot_visualizer_cfg: VisualizationMarkersCfg = POSITION_GOAL_MARKER_CFG.replace(
                    prim_path="/Visuals/PointCloud/points"
                )
                if "color" in cfg.params["debug_vis_args"]:
                    dot_visualizer_cfg.markers["target_far"].visual_material.diffuse_color = cfg.params[
                        "debug_vis_args"
                    ]["color"]
                self.dot_visualizer = VisualizationMarkers(dot_visualizer_cfg)
                self.dot_visualizer.set_visibility(True)
            else:
                rerun.init("partial-pointcloud", spawn=True)

    def __call__(
        self,
        env: ManagerBasedEnv,
        # Ray caster camera cfg
        sensor_cfg: SceneEntityCfg = SceneEntityCfg("raycaster_camera"),
        noises: list[NoiseCfg] = [],
        is_ray_caster: bool = False,
        # Point cloud cfg
        num_points: int = None,
        channel_first: bool = False,
        # Debug vis cfg
        debug_vis: bool = False,
        debug_vis_args: dict[str, Any] = {
            "env_ids": slice(None),
            "native_visualizer": False,
            "visualize_point_cloud": False,
            "visualize_depth_map": False,
            "depth_map_desired_height": 800,
            "depth_map_desired_width": 800,
        },
    ) -> torch.Tensor:
        if debug_vis:
            env_ids = debug_vis_args.get("env_ids", slice(None))
            native_visualizer = debug_vis_args.get("native_visualizer", False)
            depth_map_desired_height = debug_vis_args.get("depth_map_desired_height", 800)
            depth_map_desired_width = debug_vis_args.get("depth_map_desired_width", 800)
            visualize_point_cloud = debug_vis_args.get("visualize_point_cloud", False)
            visualize_depth_map = debug_vis_args.get("visualize_depth_map", False)

        camera: RayCasterCamera = env.scene.sensors[sensor_cfg.name]

        if is_ray_caster:
            # Point cloud in world frame
            point_clouds = camera.data.ray_hits_w.clone()

            # Get inverse quaternion
            inv_quat = math_utils.quat_inv(camera.data.quat_w)
        else:
            # Point cloud in world frame
            point_clouds = camera.ray_hits_w.clone()

            # Get inverse quaternion
            inv_quat = math_utils.quat_inv(camera.data.quat_w_world)

        # Translate point cloud to camera frame
        point_clouds -= camera.data.pos_w.unsqueeze(1)

        # Rotate point cloud to world frame
        point_clouds = math_utils.quat_rotate(inv_quat.unsqueeze(1), point_clouds)

        # Sample valid points from point cloud
        point_clouds = mesh_utils.sample_valid_points(point_clouds, num_points)

        # Apply noise to point clouds
        for noise in noises or []:
            point_clouds = noise.func(point_clouds, noise)

        if debug_vis and visualize_depth_map:
            if not is_ray_caster:
                images = camera.data.output["distance_to_image_plane"]

                images[torch.isnan(images) | torch.isinf(images)] = 5.0

                for i, depth_map in enumerate(images[env_ids]):
                    depth_map = depth_map.squeeze(0).squeeze(-1).cpu().numpy()
                    visualization_utils.show_heatmap(
                        depth_map, depth_map_desired_width, depth_map_desired_height, f"Depth Map {i}"
                    )
            else:
                print("[WARNING] Depth map visualization is only supported for ray caster camera, not ray caster")

        # Visualize point cloud if debug vis is enabled
        if debug_vis and visualize_point_cloud:
            if native_visualizer:
                self.dot_visualizer.visualize(point_clouds[env_ids].flatten(0, 1).cpu().numpy())
            else:
                rerun.log(
                    "pointcloud",
                    rerun.Points3D(
                        point_clouds[env_ids].flatten(0, 1).cpu().numpy(),
                        colors=[255, 0, 255],
                        radii=0.005,
                    ),
                )

        if channel_first:
            return point_clouds.transpose(1, 2).contiguous()
        return point_clouds
