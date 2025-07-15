import numpy as np
import pxr
import torch
import trimesh
from isaacsim.core.utils.stage import get_current_stage
from pxr import UsdGeom


def list_all_prims_and_meshes():
    """
    Lists all prims in the stage and checks which ones are meshes.
    """
    stage = get_current_stage()
    mesh_prims = []

    print("Listing all prims in the scene:")
    for prim in stage.Traverse():
        prim_path = prim.GetPath().pathString
        prim_type = prim.GetTypeName()
        print(f"   - {prim_path} ({prim_type})")

        # Check if the prim is a UsdGeom.Mesh
        if prim.IsA(UsdGeom.Mesh):
            mesh_prims.append(prim_path)

    print("\nFound Mesh Prims:")
    if mesh_prims:
        for mesh in mesh_prims:
            print(f"{mesh} (Mesh)")
    else:
        print("No `UsdGeom.Mesh` found in the scene.")


def get_point_cloud_from_mesh(prim: pxr.Usd.Prim, num_points: int = 1000) -> np.ndarray:
    """
    Extracts a point cloud from a mesh prim in the USD stage.

    Args:
        prim (pxr.UsdGeom.Mesh): The USD prim of the mesh.
        num_points (int): Number of points to sample.

    Returns:
        np.ndarray: Sampled point cloud (Nx3) or None if sampling fails.
    """

    if not prim:
        print(f"Prim '{prim}' not found.")
        return None

    mesh = UsdGeom.Mesh(prim)
    if not mesh:
        print(f"Prim '{prim}' is not a mesh.")
        return None

    # Get vertex positions
    points_attr = mesh.GetPointsAttr()
    vertices = points_attr.Get()

    # Get face indices
    face_indices_attr = mesh.GetFaceVertexIndicesAttr()
    faces = face_indices_attr.Get()

    # Check if mesh has valid data
    if vertices is None or len(vertices) == 0:
        print(f"[ERROR] Mesh at '{prim}' has no vertices.")
        return None

    if faces is None or len(faces) == 0:
        print(f"[WARNING] Mesh at '{prim}' has vertices but NO faces. Sampling from vertices instead.")
        idx = np.random.choice(vertices.shape[0], min(num_points, len(vertices)), replace=False)
        return vertices[idx]
    faces = np.array(faces).reshape(-1, 3)

    # Create a trimesh object
    mesh_trimesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # Sample points from the mesh surface
    return mesh_trimesh.sample(num_points)


def sample_valid_points(point_clouds: torch.Tensor, num_points: int):
    """
    Samples a fixed number of valid points from a given point cloud, ensuring that
    invalid points (NaN or Inf) are excluded from selection.

    If all points in a point cloud are invalid, all points in a point cloud are set to 0.0.

    Args:
        point_clouds (torch.Tensor): A batch of point clouds (num_envs, num_points, dim).
        num_points (int): The number of valid points to sample per point cloud.

    Returns:
        torch.Tensor: A tensor containing the sampled valid points with shape
                      (num_envs, num_points, dim).
    """
    valid_pc_mask = ~((torch.isnan(point_clouds) | torch.isinf(point_clouds)).any(-1))

    # Get number of valid points in each env. (num_envs, 1)
    valid_pc_counts = valid_pc_mask.sum(dim=1, keepdim=True)

    # If all points are invalid, then put one dummy point at first index of point cloud
    all_invalid_pc_mask = (valid_pc_counts == 0).any(-1)
    point_clouds[all_invalid_pc_mask, 0] = 0.0  # Set first point to 0.0
    valid_pc_mask[all_invalid_pc_mask, 0] = True  # set first point to valid

    # Sample only valid points
    valid_indices = torch.multinomial(valid_pc_mask.float(), num_samples=num_points, replacement=True)
    point_clouds = torch.gather(
        point_clouds, dim=1, index=valid_indices.unsqueeze(-1).expand(-1, -1, point_clouds.size(-1))
    )
    return point_clouds
