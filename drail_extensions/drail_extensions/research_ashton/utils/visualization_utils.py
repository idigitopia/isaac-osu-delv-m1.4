import cv2
import numpy as np
import torch
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import RAY_CASTER_MARKER_CFG


def show_heatmap(
    heatmap: torch.Tensor,
    heatmap_desired_width: int,
    heatmap_desired_height: int,
    window_name: str,
    max_val: float | None = None,
):
    assert heatmap.ndim == 2, f"Heatmap must be 2D (width, height) but got {heatmap.shape}"

    # Normalize depth map to 0-255 range and convert to uint8
    if max_val is not None:
        heatmap = (heatmap - heatmap.min()) / (max_val - heatmap.min())
    else:
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    heatmap = (heatmap * 255).astype(np.uint8)

    # color it
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_VIRIDIS)
    heatmap = cv2.resize(
        heatmap,
        (heatmap_desired_width, heatmap_desired_height),
        interpolation=cv2.INTER_LINEAR,
    )

    # Create a resizable window
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, heatmap)

    cv2.waitKey(1)


def get_point_cloud_visualizer_cfg(
    index: int = 0, color: tuple[float, float, float] = (1.0, 1.0, 1.0), radius: float = 0.002
):
    pc_visualizer_cfg: VisualizationMarkersCfg = RAY_CASTER_MARKER_CFG.replace(
        prim_path=f"/Visuals/PointCloud_{index}/points"
    )
    pc_visualizer_cfg.markers["hit"].radius = radius
    pc_visualizer_cfg.markers["hit"].visual_material.diffuse_color = color
    pc_visualizer = VisualizationMarkers(pc_visualizer_cfg)
    pc_visualizer.set_visibility(True)
    return pc_visualizer
