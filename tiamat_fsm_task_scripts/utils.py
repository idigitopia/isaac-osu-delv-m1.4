from sympy.discrete.transforms import I
import torch
import pandas as pd
import os
import cv2
import open3d as o3d
import numpy as np
import time
from scipy.spatial.transform import Rotation as R
from dataclasses import dataclass
from typing import List, Optional, Tuple

################################################################################
# Mapping Utilities
################################################################################

def get_o3d_cam_intrinsic():

    width = 1280
    height = 720
    fx = 617.3
    fy = 551.5
    ppx = 1280 / 2
    ppy = 720 / 2

    return o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, ppx, ppy)


def get_world_coordinates_from_depth(depth_tensor:torch.Tensor, camera_pose: np.ndarray, camera_intrinsics: o3d.camera.PinholeCameraIntrinsic = get_o3d_cam_intrinsic(), rgb_tensor:np.ndarray=None):
    """
    Args:
        depth_tensor: torch.Tensor of shape (W,H)
        camera_intrinsics: o3d.camera.PinholeCameraIntrinsic
        camera_pose: torch.Tensor of shape (,7), quaternion in ROS convention, and euler position in the order of [x, y, z]
    """

    # depth tensor is a numpy array of shape (720, 1280), rgb tensor is a numpy array of shape (720, 1280, 3)
    depth_tensor = depth_tensor.numpy()
    rgb_tensor = np.zeros((depth_tensor.shape[0], depth_tensor.shape[1], 3)) if rgb_tensor is None else rgb_tensor
    rgb_tensor = rgb_tensor.astype(np.uint8)

    # get tensor into o3d format
    curr_depth_img = o3d.geometry.Image(depth_tensor) # (720, 1280)
    curr_rgb_img = o3d.geometry.Image(rgb_tensor) # (720, 1280, 3)

    # calculate the local to world transformation matrix
    quat = camera_pose[[1,2,3,0]].tolist()
    position = np.array(camera_pose[4:], dtype=np.float64)
    c2w = R.from_quat(quat).as_matrix()  
    c2w = np.hstack((c2w, position.reshape(3, 1)))
    c2w = np.vstack((c2w, np.array([[0, 0, 0, 1]])))
    
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        curr_rgb_img,
        curr_depth_img,
        depth_scale=1.0,
        depth_trunc=100,
        convert_rgb_to_intensity=False
    )
    temp = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, camera_intrinsics
    )
    temp.transform(c2w)    

    return temp

################################################################################
# Object Detection Utilities
################################################################################




