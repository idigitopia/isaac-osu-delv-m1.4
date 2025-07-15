import pandas as pd
import ast
import math
import torch
from utils import get_world_coordinates_from_depth

import numpy as np
import open3d as o3d
import cv2
import json


# Parse string arrays from CSV - handle both comma-separated strings and actual lists
def safe_parse_csv_field(field_value):
    """Safely parse CSV field that could be empty, comma-separated string, or list"""
    if pd.isna(field_value) or field_value == '' or field_value == 'nan':
        return []
    
    # If it's already a list, return it
    if isinstance(field_value, list):
        return field_value
    
    # Convert to string and handle comma-separated values
    field_str = str(field_value).strip()
    if field_str == '':
        return []
    
    # Try to parse as literal first
    try:
        return ast.literal_eval(field_str)
    except (ValueError, SyntaxError):
        # If that fails, treat as comma-separated string
        if ',' in field_str:
            return [item.strip().strip("'\"") for item in field_str.split(',')]
        else:
            return [field_str.strip().strip("'\"")]



def navigation(csv_file_path):


        ## DEBUGGING CODE #########################################################
    all_points, all_colors = [], []
    pcd = o3d.geometry.PointCloud()
    #vis = o3d.visualization.Visualizer()
    #vis.create_window()
    #############################################################################

    navigation_instructions = []
    object_navigation_map = {}  


    df = pd.read_csv(csv_file_path)

    for ii in range(len(df)):

        try:
            # just sticking to first row for now. 
            first_row = df.iloc[ii]
            
            image_id = first_row['image_index']
            
            print(f"Processing image_index: {image_id}")
            print(f"Robot rotation when captured: {first_row['rotation_degrees']}°")
            print(f"Number of objects detected: {first_row['num_objects']}")
            
            # Debug: Print raw CSV values
            print(f"Raw object_ids: {repr(first_row['object_ids'])}")
            print(f"Raw centroids_x: {repr(first_row['centroids_x'])}")
            print(f"Raw centroids_y: {repr(first_row['centroids_y'])}")
            print(f"Raw depths: {repr(first_row['depths_at_centroids'])}")
            print(f"Raw object_descriptions: {repr(first_row['object_descriptions'])}")
            
            object_ids = safe_parse_csv_field(first_row['object_ids'])
            centroids_x = safe_parse_csv_field(first_row['centroids_x'])
            centroids_y = safe_parse_csv_field(first_row['centroids_y'])
            depths = safe_parse_csv_field(first_row['depths_at_centroids'])
            object_descriptions = safe_parse_csv_field(first_row['object_descriptions'])
            
            # Convert string numbers to actual numbers for centroids and depths
            try:
                centroids_x = [int(x) if isinstance(x, str) else x for x in centroids_x]
                centroids_y = [int(y) if isinstance(y, str) else y for y in centroids_y]
                depths = [float(d) if isinstance(d, str) else d for d in depths]
            except (ValueError, TypeError) as e:
                print(f"Error converting numeric values: {e}")


            # # Debug: 
            # print(f"Parsed object_ids: {object_ids}")
            # print(f"Parsed centroids_x: {centroids_x}")
            # print(f"Parsed centroids_y: {centroids_y}")
            # print(f"Parsed depths: {depths}")
            
            if not centroids_x:
                print("No objects detected in the image.")
                continue

            
        # depth_tensor = torch.load(first_row['depth_path']).squeeze(0).squeeze(0)
            # robot_pose = torch.tensor(np.concatenate((first_row['quat'], first_row['position']), axis=0))
            # world_point_cloud = get_world_coordinates_from_depth(de, first_row['pose'])

            depth_tensor = torch.load(first_row['depth_file']).squeeze(0).squeeze(0)
            rgb_tensor = torch.load(first_row['rgb_file']).squeeze(0).permute(1,2,0).numpy()

            # clean this later.
            pose_tensor = torch.load(first_row['pose_file'])
            data = pose_tensor.cpu().numpy()
            if len(data.shape) == 2:
                data = data[0]
            position, quat = data[:3].tolist(), data[3:7].tolist()
            camera_pose = np.concatenate((quat, position), axis=0)

            # DEBUGGING CODE #########################################################
            # world_point_cloud = get_world_coordinates_from_depth(depth_tensor, camera_pose = camera_pose, rgb_tensor=rgb_tensor)
            # all_points.append(np.asarray(world_point_cloud.points))
            # all_colors.append(np.asarray(world_point_cloud.colors))
                
            # pcd.points = o3d.utility.Vector3dVector(np.vstack(all_points))
            # pcd.colors = o3d.utility.Vector3dVector(np.vstack(all_colors))

            # if ii == 0:
            #     vis.add_geometry(pcd) 
            # else:
            #     vis.update_geometry(pcd)
            #########################################################################

            for i in range(len(centroids_x)):
                pixel_x = centroids_x[i]
                pixel_y = centroids_y[i]
                depth = depths[i]
                object_id = object_ids[i] 
                
                # Get the corresponding object description
                object_description = object_descriptions[i] if i < len(object_descriptions) else f"object_{object_id}"
                
                print(f"\nObject ID '{object_id}' ({object_description}):")
                print(f"  Pixel coordinates: ({pixel_x}, {pixel_y})")
                print(f"  Depth: {depth}m")

                world_heading_degrees = 0
                new_depth_tensor = depth_tensor.clone()
                new_depth_tensor = torch.ones_like(new_depth_tensor) * 1e9
                new_depth_tensor[pixel_y-4:pixel_y+4, pixel_x-4:pixel_x+4] = depth_tensor[pixel_y-4:pixel_y+4, pixel_x-4:pixel_x+4]

                world_point_cloud = get_world_coordinates_from_depth(new_depth_tensor, camera_pose, rgb_tensor=None)

                # this is getting 
                world_x, world_y, world_z = float(world_point_cloud.points[0][0]), float(world_point_cloud.points[0][1]), float(world_point_cloud.points[0][2])

                # need to convert types to native python types for json serialization
                navigation_instruction = [float(world_x), float(world_y), float(world_z), float(world_heading_degrees), str(object_id), int(image_id)]
                navigation_instructions.append(navigation_instruction)
                
                
                object_navigation_map[object_description] = {
                    "navigation": navigation_instruction,
                    "object_id": str(object_id),  
                    "image_id": int(image_id),   
                }
                
        except Exception as e:
            print(f"Error: {e}")
            continue
    
    #print(f"Navigation instructions: {navigation_instructions}")
    
    # save mapping as json
    json_output_path = csv_file_path.replace('scan_metadata_with_objects.csv', 'object_navigation_map.json')
    with open(json_output_path, 'w') as f:
        json.dump(object_navigation_map, f, indent=2)
    
    print(f"Object navigation map saved to: {json_output_path}")
    
    return navigation_instructions, json_output_path

# if __name__ == "__main__":
#     navigation_instructions, object_map = navigation("tiamat_fsm_task_scripts/data/scan_data/scan_metadata_with_objects.csv")

# def navigation(csv_file_path):
#     """
#     Process the first row of the CSV file to generate navigation instructions.
    
#     Args:
#         csv_file_path: Path to the CSV file containing detection data
        
#     Returns:
#         navigation_instructions: List of [x, y, z, heading_degrees, object_id, image_id] for each detected object
#     """
    
#     navigation_instructions = []
    
#     df = pd.read_csv(csv_file_path)

#     for i in range(len(df)):
#         # just sticking to first row for now. 
#         if i > 0:
#             continue
#         first_row = df.iloc[i]
        
#         image_id = first_row['image_index']
        
#         print(f"Processing image_index: {image_id}")
#         print(f"Robot rotation when captured: {first_row['rotation_degrees']}°")
#         print(f"Number of objects detected: {first_row['num_objects']}")
        
#         # Debug: Print raw CSV values
#         print(f"Raw object_ids: {repr(first_row['object_ids'])}")
#         print(f"Raw centroids_x: {repr(first_row['centroids_x'])}")
#         print(f"Raw centroids_y: {repr(first_row['centroids_y'])}")
#         print(f"Raw depths: {repr(first_row['depths_at_centroids'])}")
        
#         object_ids = safe_parse_csv_field(first_row['object_ids'])
#         centroids_x = safe_parse_csv_field(first_row['centroids_x'])
#         centroids_y = safe_parse_csv_field(first_row['centroids_y'])
#         depths = safe_parse_csv_field(first_row['depths_at_centroids'])
        
#         # Convert string numbers to actual numbers for centroids and depths
#         try:
#             centroids_x = [int(x) if isinstance(x, str) else x for x in centroids_x]
#             centroids_y = [int(y) if isinstance(y, str) else y for y in centroids_y]
#             depths = [float(d) if isinstance(d, str) else d for d in depths]
#         except (ValueError, TypeError) as e:
#             print(f"Error converting numeric values: {e}")
#             return []
    
#         # # Debug: 
#         # print(f"Parsed object_ids: {object_ids}")
#         # print(f"Parsed centroids_x: {centroids_x}")
#         # print(f"Parsed centroids_y: {centroids_y}")
#         # print(f"Parsed depths: {depths}")
        
#         if not centroids_x:
#             print("No objects detected in the image.")
#             return []
        
        
#     # depth_tensor = torch.load(first_row['depth_path']).squeeze(0).squeeze(0)
#         # robot_pose = torch.tensor(np.concatenate((first_row['quat'], first_row['position']), axis=0))
#         # world_point_cloud = get_world_coordinates_from_depth(de, first_row['pose'])

#         depth_tensor = torch.load(first_row['depth_file']).squeeze(0).squeeze(0)
#         rgb_tensor = cv2.imread(first_row['rgb_file'])

#         # clean this later.
#         pose_tensor = torch.load(first_row['pose_file'])
#         data = pose_tensor.cpu().numpy()
#         if len(data.shape) == 2:
#             data = data[0]
#         position, quat = data[:3].tolist(), data[3:7].tolist()
#         camera_pose = np.concatenate((quat, position), axis=0)

#         robot_position = [0.0, 0.0, 0.0] #robot always starts at origin
#         robot_rotation_degrees = first_row['rotation_degrees']
        
        
#         image_width = 1280
#         image_center_x = image_width / 2  # 640
#         horizontal_fov_degrees = 60
        
        
    
#         for i in range(len(centroids_x)):
#             pixel_x = centroids_x[i]
#             pixel_y = centroids_y[i]
#             depth = depths[i]
#             object_id = object_ids[i] 
            
#             print(f"\nObject ID '{object_id}':")
#             print(f"  Pixel coordinates: ({pixel_x}, {pixel_y})")
#             print(f"  Depth: {depth}m")
            
#             # calculate angular offset from bottom center of image
#             angular_offset_degrees = (pixel_x - image_center_x) / image_width * horizontal_fov_degrees
            
            
#             world_heading_degrees = robot_rotation_degrees + angular_offset_degrees
            
            
#             world_heading_radians = math.radians(world_heading_degrees)
            
            
#             world_x = robot_position[0] + depth * math.cos(world_heading_radians)
#             world_y = robot_position[1] + depth * math.sin(world_heading_radians)
#             world_z = 0.0
            
#             print(f"  Angular offset: {angular_offset_degrees:.6f}°")
#             print(f"  World heading: {world_heading_degrees:.6f}°")
#             print(f"  World coordinates: ({world_x:.2f}, {world_y:.2f}, {world_z:.2f})")
            
#             # Add to navigation instructions with the actual object ID and image ID
#             world_point_cloud = get_world_coordinates_from_depth(depth_tensor[centroids_x[i], centroids_y[i]], camera_pose, rgb_tensor=rgb_tensor)
#             world_x, world_y, world_z = world_point_cloud.points[0], world_point_cloud.points[1], world_point_cloud.points[2]

#             print(f"  World point cloud: {world_point_cloud}")

#             navigation_instructions.append([world_x, world_y, world_z, world_heading_degrees, object_id, image_id])
        
#         print(f"\nGenerated {len(navigation_instructions)} navigation instructions:")
#         for i, instruction in enumerate(navigation_instructions):
#             print(f"  Target {i+1}: [{instruction[0]:.2f}, {instruction[1]:.2f}, {instruction[2]:.2f}, {instruction[3]:.2f}°, ID={instruction[4]}, IMG={instruction[5]}]")
    
#     return navigation_instructions


