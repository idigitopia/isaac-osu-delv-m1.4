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


# checklist:
    # [x] N images: 13 
    # [x] N camera poses (quats) for each image
         # - saved in data/scan_data/pose
    # [x] Camera intrinsic parameters

# known:
# ALL_BOX_POSITIONS = [
#    (-4.0, 5.0, 2.0),    # Box 0 (dark gray) 128.66°
#    (3.0, 2.0, 2.0),     # Box 1 (red) 33.69°
#    (2.97, -2.97, 2.0),  # Box 2 (green) -45.00°
#    (-2.97, -2.97, 2.0), # Box 3 (blue) -135.00°
#    (-2.97, 2.97, 2.0),  # Box 4 (yellow) 135.00°
#    (2.5, 2.5, 2.0),     # Box 5 (magenta) 45.00°
#    (2.5, -2.5, 2.0),    # Box 6 (cyan) -45.00°
#    (-3.5, -2.0, 2.0),   # Box 7 (orange) -150.26°
# ]

@dataclass
class CameraIntrinsics:
    focal_length: float = 1.88
    focus_distance: float = 0.5
    horizontal_aperture: float = 3.896
    vertical_aperture: float = 2.453
    clipping_range: Tuple[float, float] = (0.01, 6.0)
    width: int = 1280
    height: int = 720

# each image contains the following fields.
@dataclass
class ScanImage:
    index: int
    angle: float
    position: List[float]
    quat: List[float]
    num_objects: int
    centroid_x: List[float]
    centroid_y: List[float]
    depths_at_centroids: List[float]
    image_path: str
    depth_path: str
    pose_path: str

def parse_csv_list(value):
    if pd.isna(value) or value == "":
        return []
    return [float(x.strip()) for x in str(value).split(",")]

def load_pose(pose_file: str):
    pose_tensor = torch.load(pose_file)
    data = pose_tensor.cpu().numpy()
    if len(data.shape) == 2:
        data = data[0]
    return data[:3].tolist(), data[3:7].tolist()

def create_data():
    csv_data = pd.read_csv("tiamat_fsm_task_scripts/data/scan_data/scan_metadata_with_objects.csv")
    images = []
    
    for i in range(13):
        row = csv_data[csv_data['image_index'] == i].iloc[0]
        pose_file = f"tiamat_fsm_task_scripts/data/scan_data/pose/pose_{i:03d}.pt"
        position, quat = load_pose(pose_file)
        
        images.append(ScanImage(
            index=i,
            angle=row['rotation_degrees'],
            position=position,
            quat=quat,
            num_objects=row['num_objects'],
            centroid_x=parse_csv_list(row['centroids_x']),
            centroid_y=parse_csv_list(row['centroids_y']),
            depths_at_centroids=parse_csv_list(row['depths_at_centroids']),
            image_path=f"tiamat_fsm_task_scripts/data/bounded_imgs/bounded_imgs_{i:03d}.png",
            depth_path=f"tiamat_fsm_task_scripts/data/scan_data/depth/depth_{i:03d}.pt",
            pose_path=pose_file
        ))
    
    return images

#instantiate data.
camera = CameraIntrinsics()
images = create_data()


def convert_to_world(scan_image: ScanImage, pixel_x: float, pixel_y: float, depth: float):
    """Convert pixel coordinates to world coordinates using camera pose and intrinsics"""
    # TODO: Implement pixel to world coordinate conversion
        # Chanho will save me
    
    
    pass

# @dataclass
# class CameraIntrinsics:
#     focal_length: float = 1.88
#     focus_distance: float = 0.5
#     horizontal_aperture: float = 3.896
#     vertical_aperture: float = 2.453
#     clipping_range: Tuple[float, float] = (0.01, 6.0)
#     width: int = 1280
#     height: int = 720

def get_o3d_cam_intrinsic():

    width = 1280
    height = 720
    fx = 617.3
    fy = 551.5
    ppx = 1280 / 2
    ppy = 720 / 2

    return o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, ppx, ppy)

# print them all
all_points = []
all_colors = []
pcd = o3d.geometry.PointCloud() #pcd probably stands for point cloud
vis = o3d.visualization.Visualizer()
vis.create_window()
is_first = True

for i in range(13):
    if i == 9:
        continue
    print(i)
    img = images[i]
    
    depth_tensor = torch.load(img.depth_path) 
    depth_tensor = depth_tensor.squeeze(0).squeeze(0) # (1,1,720,1280) -> (720, 1280) 
    depth_tensor = depth_tensor.numpy()

    curr_depth = o3d.geometry.Image(depth_tensor) # (720, 1280)
    # curr_rgb_img = o3d.io.read_image(img.image_path)
    
    # NOTE:
        # Currently reads the already processed bounded images, rather than the raw RGB images. 
        # Using the raw RGB images would be more robust, as there is stochasticity in the bounding box generation (some images' dimensions are slightly elongated).
    
    curr_rgb_img = cv2.imread(img.image_path)    
    curr_rgb_img = o3d.geometry.Image(curr_rgb_img)
    
        # curr_rgb_img = o3d.geometry.Image(curr_rgb_img)
    # import pdb; pdb.set_trace()

    quat = np.array(img.quat, dtype=np.float64)
    position = np.array(img.position, dtype=np.float64)

    # confirmed that the quaternion is in the order of [x, y, z, w]
    quat = quat[[1,2,3,0]].tolist()
    # quat = quat.tolist()
    w2c = R.from_quat(quat).as_matrix()  
    
    w2c = np.hstack((w2c, position.reshape(3, 1)))
    w2c = np.vstack((w2c, np.array([[0, 0, 0, 1]])))
    
    # define a 4x4 transformation matrix as a numpy array
    # cam_trans = np.eye(4)
    # cam_trans[1, 1] = -1
    # cam_trans[2, 2] = -1

    # TODO: we aren't sure which one is correct yet (w2c or c2w)
    # c2w = np.linalg.inv(w2c)
    c2w = w2c
    # rot_object = R.from_quat(quat).inv()

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        curr_rgb_img,
        curr_depth,
        depth_scale=1.0,
        depth_trunc=100,
        convert_rgb_to_intensity=False
    )
    temp = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, get_o3d_cam_intrinsic()
    )
    # import pdb; pdb.set_trace()
    temp.transform(c2w)    

    all_points.append(np.asarray(temp.points))
    all_colors.append(np.asarray(temp.colors))
        
    pcd.points = o3d.utility.Vector3dVector(np.vstack(all_points))
    pcd.colors = o3d.utility.Vector3dVector(np.vstack(all_colors))

    if i == 0 and is_first:
        vis.add_geometry(pcd) 
        is_first = False

    vis.update_geometry(pcd)

while True:
    vis.poll_events()
    vis.update_renderer()
    # vis.capture_screen_image(os.path.join(out_dir, "%05d.jpg" % frame))
    time.sleep(0.01)

# NOTE:
    # code takes depth data, rgb data, and camera pose data, and converts it to a point cloud in ROS coordinate system.
    # need to convert it into Isaac coordinate system.
