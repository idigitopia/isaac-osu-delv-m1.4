import torch
import pandas as pd
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

@dataclass
class CameraIntrinsics:
    focal_length: float = 1.88
    focus_distance: float = 0.5
    horizontal_aperture: float = 3.896
    vertical_aperture: float = 2.453
    clipping_range: Tuple[float, float] = (0.01, 6.0)
    width: int = 1280
    height: int = 720

@dataclass
class ScanImage:
    index: int
    angle: float
    position: List[float]
    rotation: List[float]
    num_objects: int
    centroid_x: List[float]
    centroid_y: List[float]
    depths_at_centroids: List[float]
    image_path: str
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
        position, rotation = load_pose(pose_file)
        
        images.append(ScanImage(
            index=i,
            angle=row['rotation_degrees'],
            position=position,
            rotation=rotation,
            num_objects=row['num_objects'],
            centroid_x=parse_csv_list(row['centroids_x']),
            centroid_y=parse_csv_list(row['centroids_y']),
            depths_at_centroids=parse_csv_list(row['depths_at_centroids']),
            image_path=f"tiamat_fsm_task_scripts/data/bounded_imgs/bounded_imgs_{i:03d}.png",
            pose_path=pose_file
        ))
    
    return images

camera_intrinsics = CameraIntrinsics()
images = create_data()

def convert_to_world(scan_image: ScanImage, pixel_x: float, pixel_y: float, depth: float):
    """Convert pixel coordinates to world coordinates using camera pose and intrinsics"""
    # TODO: Implement pixel to world coordinate conversion
    # Uses: scan_image.position, scan_image.rotation, camera_intrinsics, pixel coords, depth
    pass



# Print first 3
for i in range(3):
    img = images[i]
    print(f"Image {img.index}: {img.angle}°")
    print(f"  Position: {img.position}")
    print(f"  Quaternion: {img.rotation}")
    print(f"  Objects: {img.num_objects}")
    if img.num_objects > 0:
        for j in range(len(img.centroid_x)):
            print(f"    Object {j}: ({img.centroid_x[j]}, {img.centroid_y[j]}) depth {img.depths_at_centroids[j]}")


