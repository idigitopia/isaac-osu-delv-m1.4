"""
Object Detection Pipeline using Segment Anything Model (SAM)

This module provides functionality for detecting and analyzing objects in RGB-D images
using SAM for segmentation, followed by filtering and spatial analysis.
"""

import pandas as pd
import torch
import cv2
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os

sam_checkpoint = "tiamat_fsm_task_scripts/models/sam_vit_b_01ec64.pth"
model_type = "vit_b"
device = "cuda" if torch.cuda.is_available() else "cpu"

# check if SAM is already downloaded
if not os.path.exists(sam_checkpoint):
    print(f"Downloading SAM model to {sam_checkpoint}")
    import urllib.request
    url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    os.makedirs("models", exist_ok=True)
    print("Downloading SAM model...")
    urllib.request.urlretrieve(url, "models/sam_vit_b_01ec64.pth")
    print("Download complete!")


sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)


mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.9,
    stability_score_thresh=0.95,
    crop_n_layers=0,
    crop_n_points_downscale_factor=1,
    min_mask_region_area=100,
)

def load_and_convert_images(row):
    """
    Load RGB, depth, and pose tensors from file paths and convert to numpy arrays.
    
    Args:
        row: DataFrame row containing 'rgb_file', 'depth_file', and 'pose_file' paths
        
    Returns:
        tuple: (rgb_image, depth_image, pose_tensor) as numpy arrays/tensor
    """
    rgb_tensor = torch.load(row['rgb_file'])
    # rgb_image = rgb_tensor.numpy()
    rgb_image = rgb_tensor.squeeze(0).permute(1, 2, 0).numpy()
    # rgb_image = np.clip(rgb_image * 255, 0, 255).astype(np.uint8)
    
    depth_tensor = torch.load(row['depth_file'])
    depth_image = depth_tensor.squeeze().numpy()
    
    # Load pose tensor (7D: [x, y, z, qw, qx, qy, qz])
    pose_tensor = torch.load(row['pose_file'])
    
    return rgb_image, depth_image, pose_tensor

def get_background_brightness(image):
    """
    Estimate background brightness by sampling corner regions of the image.
    
    Args:
        image: RGB image as numpy array
        
    Returns:
        float: Average brightness value of corner regions
    """
    h, w = image.shape[:2]
    
    corner_size = min(50, h//10, w//10)
    corners = [
        image[:corner_size, :corner_size],
        image[:corner_size, -corner_size:],
        image[-corner_size:, :corner_size],
        image[-corner_size:, -corner_size:]
    ]
    
    corner_brightness = []
    for corner in corners:
        hsv_corner = cv2.cvtColor(corner, cv2.COLOR_RGB2HSV)
        corner_brightness.append(np.mean(hsv_corner[:, :, 2]))
    
    return np.mean(corner_brightness)

def passes_black_filter(image, mask, black_threshold=2, black_percentage_threshold=0.70):
    """
    Filter out objects that are predominantly black or very dark.
    
    Args:
        image: RGB image as numpy array
        mask: SAM mask dictionary with 'segmentation' key
        black_threshold: RGB value threshold for considering a pixel black
        black_percentage_threshold: Maximum allowed percentage of black pixels
        
    Returns:
        bool: True if object passes the filter (not predominantly black)
    """
    coords = np.where(mask['segmentation'])
    if len(coords[0]) == 0:
        return False
    
    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()
    
    object_region = image[y_min:y_max+1, x_min:x_max+1]
    
    # RGB channels are below threshold (truly dark pixels)?
    r_channel = object_region[:, :, 0]
    g_channel = object_region[:, :, 1] 
    b_channel = object_region[:, :, 2]
    
    black_pixels_mask = (r_channel < black_threshold) & (g_channel < black_threshold) & (b_channel < black_threshold)
    black_pixels = np.sum(black_pixels_mask)
    total_pixels = object_region.shape[0] * object_region.shape[1]
    black_percentage = black_pixels / total_pixels
    
    return black_percentage < black_percentage_threshold

def passes_color_filter(image, mask, background_brightness, saturation_threshold=30, brightness_diff_threshold=40):
    """
    Filter objects based on color saturation and brightness contrast with background.
    
    Args:
        image: RGB image as numpy array
        mask: SAM mask dictionary with 'segmentation' key
        background_brightness: Background brightness value for comparison
        saturation_threshold: Minimum saturation for colorful objects
        brightness_diff_threshold: Minimum brightness difference from background
        
    Returns:
        bool: True if object passes color filter criteria
    """
    coords = np.where(mask['segmentation'])
    if len(coords[0]) == 0:
        return False
    
    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()
    
    object_region = image[y_min:y_max+1, x_min:x_max+1]
    hsv_region = cv2.cvtColor(object_region, cv2.COLOR_RGB2HSV)
    
    mean_saturation = np.mean(hsv_region[:, :, 1])
    mean_brightness = np.mean(hsv_region[:, :, 2])
    
    
    if mean_saturation > saturation_threshold:
        return True
    
    
    brightness_diff = abs(mean_brightness - background_brightness)
    if brightness_diff > brightness_diff_threshold:
        return True
    
    return False

def extract_bounding_boxes(image, depth_image, masks, max_width=300, max_height=300, saturation_threshold=30, brightness_diff_threshold=40, black_threshold=50, black_percentage_threshold=0.7, expansion_factor=0.2):
    """
    Convert SAM masks to filtered bounding boxes based on size and appearance criteria.
    
    Args:
        image: RGB image as numpy array
        masks: List of SAM mask dictionaries
        max_width: Maximum allowed bounding box width
        max_height: Maximum allowed bounding box height
        saturation_threshold: Color saturation threshold for filtering
        brightness_diff_threshold: Brightness difference threshold for filtering
        black_threshold: RGB threshold for black pixel detection
        black_percentage_threshold: Maximum allowed percentage of black pixels
        expansion_factor: Factor to expand bounding boxes (0.2 = 20% bigger)
        
    Returns:
        list: Filtered bounding box dictionaries with expanded boundaries
    """
    bounding_boxes = []
    
    background_brightness = get_background_brightness(image)
    print(f"Background brightness: {background_brightness:.1f}")
    
    # img dims for boundary check
    img_height, img_width = image.shape[:2]
    
    for i, mask in enumerate(masks):
        coords = np.where(mask['segmentation'])
        
        if len(coords[0]) > 0:
            y_min, y_max = coords[0].min(), coords[0].max()
            x_min, x_max = coords[1].min(), coords[1].max()
            
        
            width = x_max - x_min
            height = y_max - y_min
            
            
            if width <= max_width and height <= max_height:

                if depth_image[(y_min+y_max)//2, (x_min+x_max)//2] > 5 or depth_image[(y_min+y_max)//2, (x_min+x_max)//2] < 1:
                    continue

                if width < 50 or height < 50:
                    continue

                # apply filters
                if passes_color_filter(image, mask, background_brightness, saturation_threshold, brightness_diff_threshold) and passes_black_filter(image, mask, black_threshold, black_percentage_threshold):
                    
                    
                    width_expansion = int(width * expansion_factor / 2)
                    height_expansion = int(height * expansion_factor / 2)
                    
                    
                    expanded_x_min = max(0, x_min - width_expansion)
                    expanded_y_min = max(0, y_min - height_expansion)
                    expanded_x_max = min(img_width - 1, x_max + width_expansion)
                    expanded_y_max = min(img_height - 1, y_max + height_expansion)
                    
                    
                    new_width = expanded_x_max - expanded_x_min
                    new_height = expanded_y_max - expanded_y_min
                    
                    bbox = {
                        'object_id': i,
                        'bbox': (expanded_x_min, expanded_y_min, expanded_x_max, expanded_y_max),
                        'width': new_width,
                        'height': new_height,
                        'area': mask['area'],  
                        'confidence': mask['predicted_iou']
                    }
                    bounding_boxes.append(bbox)
                    
                    print(f"Object {i}: Original bbox ({x_min}, {y_min}, {x_max}, {y_max}) -> Expanded bbox ({expanded_x_min}, {expanded_y_min}, {expanded_x_max}, {expanded_y_max})")
                else:
                    print(f"Filtered out object {i}: failed color or black filter")
            else:
                print(f"Filtered out object {i}: size ({width}x{height}) exceeds threshold ({max_width}x{max_height})")
    
    return bounding_boxes

def calculate_distance(bbox1, bbox2):
    """
    Calculate Euclidean distance between centers of two bounding boxes.
    
    Args:
        bbox1, bbox2: Bounding box dictionaries with 'bbox' key
        
    Returns:
        float: Distance between bounding box centers
    """
    x1_min, y1_min, x1_max, y1_max = bbox1['bbox']
    x2_min, y2_min, x2_max, y2_max = bbox2['bbox']
    
    center1_x, center1_y = (x1_min + x1_max) / 2, (y1_min + y1_max) / 2
    center2_x, center2_y = (x2_min + x2_max) / 2, (y2_min + y2_max) / 2
    
    distance = np.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)
    return distance

def merge_close_boxes(bboxes, distance_threshold=100):
    """
    Merge bounding boxes that are within a specified distance threshold.
    
    Args:
        bboxes: List of bounding box dictionaries
        distance_threshold: Maximum distance for merging boxes
        
    Returns:
        list: Merged bounding box dictionaries
    """
    if len(bboxes) <= 1:
        return bboxes
    
    merged_boxes = []
    used = set()
    
    for i, bbox in enumerate(bboxes):
        if i in used:
            continue
            
        close_boxes = [bbox]
        close_indices = [i]
        
        # find boxes within distance threshold
        for j, other_bbox in enumerate(bboxes):
            if j != i and j not in used:
                distance = calculate_distance(bbox, other_bbox)
                if distance <= distance_threshold:
                    close_boxes.append(other_bbox)
                    close_indices.append(j)
        
        
        for idx in close_indices:
            used.add(idx)
        
        
        all_x_mins = [b['bbox'][0] for b in close_boxes]
        all_y_mins = [b['bbox'][1] for b in close_boxes]
        all_x_maxs = [b['bbox'][2] for b in close_boxes]
        all_y_maxs = [b['bbox'][3] for b in close_boxes]
        
        merged_bbox = {
            'object_id': f"merged_{len(merged_boxes)}",
            'bbox': (min(all_x_mins), min(all_y_mins), max(all_x_maxs), max(all_y_maxs)),
            'width': max(all_x_maxs) - min(all_x_mins),
            'height': max(all_y_maxs) - min(all_y_mins),
            'area': sum([b['area'] for b in close_boxes]),
            'confidence': np.mean([b['confidence'] for b in close_boxes]),
            'original_count': len(close_boxes)
        }
        
        merged_boxes.append(merged_bbox)
    
    return merged_boxes

def save_bounded_image(image, bboxes, image_idx, rgb_file, output_dir):
    """
    Save visualization of detected objects with bounding boxes overlaid.
    Preserves original image dimensions.
    
    Args:
        image: RGB image as numpy array
        bboxes: List of bounding box dictionaries
        image_idx: Index for output filename
        rgb_file: Original RGB file path (for reference)
        output_dir: Directory to save output image
        
    Returns:
        str: Path to saved output image
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure figure to match original image dimensions
    height, width = image.shape[:2]
    dpi = 100
    plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
    plt.ioff()
    
    plt.imshow(image)
    
    # Draw bounding boxes and labels
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox['bbox']
        rect = Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, 
                           fill=False, color='red', linewidth=2)
        plt.gca().add_patch(rect)

        label_x = max(0, min(x_min, width - 50)) 
        label_y = max(15, min(y_min, height - 5))  

        plt.text(label_x, label_y, f"ID:{bbox['object_id']}", 
                color='red', fontsize=10, weight='bold')
    
    # Remove axes and padding to preserve exact dimensions
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    output_filename = f"bounded_imgs_{image_idx:03d}.png"
    output_path = os.path.join(output_dir, output_filename)
    
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    return output_path

def calculate_object_spatial_data(bboxes, depth_img):
    """
    Extract spatial data (centroids, depth values) for objects with valid depth information.
    
    Args:
        bboxes: List of bounding box dictionaries
        depth_img: Depth image as numpy array
        
    Returns:
        list: Objects with valid spatial data
    """
    valid_objects = []
    
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox['bbox']
        
        
        centroid_x = int((x_min + x_max) / 2)
        centroid_y = int((y_min + y_max) / 2)

        
        depth_at_centroid = depth_img[centroid_y, centroid_x]
        MAX_DEPTH = 6
        
        if depth_at_centroid > 0  and depth_at_centroid < MAX_DEPTH and np.isfinite(depth_at_centroid):
            valid_objects.append({
                'object_id': bbox['object_id'],
                'centroid_x': centroid_x,
                'centroid_y': centroid_y,
                'depth': float(depth_at_centroid)
            })
        else:
            print(f"Filtered out object {bbox['object_id']} at ({centroid_x}, {centroid_y}): invalid depth {depth_at_centroid}")
        
    return valid_objects

def format_object_data_for_csv(valid_objects):
    """
    Convert object data into CSV-compatible format with comma-separated values.
    
    Args:
        valid_objects: List of object dictionaries with spatial data
        
    Returns:
        dict: Formatted data ready for CSV export
    """
    if not valid_objects:
        return {
            'num_objects': 0,
            'object_ids': '',
            'centroids_x': '',
            'centroids_y': '',
            'depths_at_centroids': ''
        }
    
    
    object_ids = [str(obj['object_id']) for obj in valid_objects]
    centroids_x = [str(obj['centroid_x']) for obj in valid_objects]
    centroids_y = [str(obj['centroid_y']) for obj in valid_objects]
    depths = [str(obj['depth']) for obj in valid_objects]
    
    return {
        'num_objects': len(valid_objects),
        'object_ids': ','.join(object_ids),
        'centroids_x': ','.join(centroids_x),
        'centroids_y': ','.join(centroids_y),
        'depths_at_centroids': ','.join(depths)
    }

def process_single_image(row, image_idx, output_dir, max_width=300, max_height=300, distance_threshold=65, saturation_threshold=30, brightness_diff_threshold=40, black_threshold=50, black_percentage_threshold=0.7, expansion_factor=0.2):
    """
    Process a single image through the complete detection pipeline.
    
    Args:
        row: DataFrame row containing image file paths
        image_idx: Index of the image being processed
        output_dir: Directory for saving output visualizations
        expansion_factor: Factor to expand bounding boxes (0.2 = 20% bigger)
        Additional args: Various threshold parameters for filtering
        
    Returns:
        dict: Processing results including success status and object data
    """
    try:
        print(f"\n--- Processing Image {image_idx} ---")
        print(f"RGB file: {row['rgb_file']}")
        
        
        rgb_img, depth_img, pose_tensor = load_and_convert_images(row)
        print(f"RGB shape: {rgb_img.shape}, Depth shape: {depth_img.shape}, Pose shape: {pose_tensor.shape}")
        
        
        detected_masks = mask_generator.generate(rgb_img)
        print(f"Found {len(detected_masks)} masks")
        
       
        bboxes = extract_bounding_boxes(rgb_img, depth_img, detected_masks, max_width=max_width, max_height=max_height, 
                                       saturation_threshold=saturation_threshold, brightness_diff_threshold=brightness_diff_threshold,
                                       black_threshold=black_threshold, black_percentage_threshold=black_percentage_threshold,
                                       expansion_factor=expansion_factor)
        print(f"Extracted {len(bboxes)} bounding boxes (after size, color, and black filtering)")
        
        # merge nearby boxes
        merged_bboxes = merge_close_boxes(bboxes, distance_threshold=distance_threshold)
        print(f"Final count: {len(merged_bboxes)} bounding boxes (after merging)")
        
        #  spatial data
        valid_objects = calculate_object_spatial_data(merged_bboxes, depth_img)
        print(f"Valid objects with depth: {len(valid_objects)}")
        
        csv_data = format_object_data_for_csv(valid_objects)
        
        
        output_path = save_bounded_image(rgb_img, merged_bboxes, image_idx, row['rgb_file'], output_dir)
        print(f"Saved bounded image to: {output_path}")
        
        
        result = {
            'image_idx': image_idx,
            'output_path': output_path,
            'total_masks': len(detected_masks),
            'filtered_boxes': len(bboxes),
            'final_boxes': len(merged_bboxes),
            'success': True
        }
        result.update(csv_data)
        
        return result
        
    except Exception as e:
        print(f"Error processing image {image_idx}: {e}")
        empty_csv_data = format_object_data_for_csv([])
        result = {
            'image_idx': image_idx,
            'error': str(e),
            'success': False
        }
        result.update(empty_csv_data)
        
        return result

def detection(csv_file_path='tiamat_fsm_task_scripts/data/scan_data/scan_metadata.csv', 
         output_dir="tiamat_fsm_task_scripts/data/bounded_imgs", 
         max_width=300, max_height=300, distance_threshold=50, 
         saturation_threshold=30, brightness_diff_threshold=40, expansion_factor=0.2):
    """
    Main function to process all images in batch and update CSV with detection results.
    
    Args:
        csv_file_path: Path to input CSV containing image metadata
        output_dir: Directory for saving output visualizations
        max_width/max_height: Size thresholds for object filtering
        distance_threshold: Distance threshold for merging nearby objects
        saturation_threshold: Color saturation threshold for filtering
        brightness_diff_threshold: Brightness difference threshold for filtering
        expansion_factor: Factor to expand bounding boxes (0.2 = 20% bigger)
        
    Returns:
        tuple: (processing_results, output_csv_path)
    """
    print("=== SAM Batch Processing with HSV Filtering ===")
    print(f"CSV file: {csv_file_path}")
    print(f"Output directory: {output_dir}")
    print(f"Size threshold: {max_width}x{max_height}")
    print(f"Distance threshold: {distance_threshold}")
    print(f"Saturation threshold: {saturation_threshold}")
    print(f"Brightness difference threshold: {brightness_diff_threshold}")
    print(f"Bounding box expansion factor: {expansion_factor} ({expansion_factor*100}% bigger)")
    print(f"Device: {device}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    
    df = pd.read_csv(csv_file_path)
    print(f"Loaded metadata with {len(df)} entries")
    print(f"Processing ALL {len(df)} images...")
    
    
    df['num_objects'] = 0
    df['object_ids'] = ''  
    df['centroids_x'] = ''
    df['centroids_y'] = ''
    df['depths_at_centroids'] = ''
    
    results = []
    
    
    for i in range(len(df)):
        row = df.iloc[i]
        result = process_single_image(row, i, output_dir, max_width, max_height, 
                                    distance_threshold, saturation_threshold, brightness_diff_threshold,
                                    black_threshold=50, black_percentage_threshold=0.7,
                                    expansion_factor=expansion_factor)
        results.append(result)
        
        
        if result['success']:
            df.loc[i, 'num_objects'] = result['num_objects']
            df.loc[i, 'object_ids'] = result['object_ids']
            df.loc[i, 'centroids_x'] = result['centroids_x']
            df.loc[i, 'centroids_y'] = result['centroids_y']
            df.loc[i, 'depths_at_centroids'] = result['depths_at_centroids']
        
    
    output_csv_path = csv_file_path.replace('.csv', '_with_objects.csv')
    df.to_csv(output_csv_path, index=False)
    print(f"\nUpdated CSV saved to: {output_csv_path}")
    
   
    print("\n=== BATCH PROCESSING SUMMARY ===")
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"Successfully processed: {len(successful)}/{len(df)}")
    print(f"Failed: {len(failed)}/{len(df)}")
    print(f"Images saved to: {output_dir}")
    
    if successful:
        total_masks = sum(r['total_masks'] for r in successful)
        total_final_boxes = sum(r['final_boxes'] for r in successful)
        total_objects = sum(r['num_objects'] for r in successful)
        avg_objects_per_image = total_objects / len(successful)
        
        print(f"Total masks detected: {total_masks}")
        print(f"Total final bounding boxes: {total_final_boxes}")
        print(f"Total valid objects: {total_objects}")
        print(f"Average objects per image: {avg_objects_per_image:.2f}")
    
    if failed:
        print("\nFailed images:")
        for r in failed:
            print(f"  Image {r['image_idx']}: {r['error']}")
    
    return results, output_csv_path