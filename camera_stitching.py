import cv2
import numpy as np
import json
import time
from typing import Dict, Tuple
# from mock_camera import poll_frame_data  # for testing without pyrealsense cameras , using local video file streams instead
from camera import poll_frame_data
from image import sharpen_and_rotate_image, buffer_to_array

def load_calibration_markers(file_path: str) -> Dict:
    """Load calibration markers from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def calculate_perspective_transform(markers: Dict) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Calculate perspective transform matrix from calibration markers
    
    Args:
        markers: Dictionary containing calibration markers with pixel and physical positions
    
    Returns:
        transform_matrix: 3x3 perspective transform matrix
        output_size: (width, height) of the output image in pixels
    """
    # Extract pixel positions (source points)
    src_points = []
    physical_points = []
    
    # Order: top_left, top_right, bottom_right, bottom_left
    for position in ["top_left", "top_right", "bottom_right", "bottom_left"]:
        marker = markers["calibration_markers"][position]
        src_points.append(marker["pixel_position"])
        physical_points.append(marker["physical_position"])
    
    src_points = np.array(src_points, dtype=np.float32)
    physical_points = np.array(physical_points, dtype=np.float32)
    
    # Calculate physical dimensions (in cm) - this is just between markers
    marker_width = max(physical_points[:, 0]) - min(physical_points[:, 0])
    marker_height = max(physical_points[:, 1]) - min(physical_points[:, 1])
    
    # Add 3cm on each side to get the full table dimensions
    # Markers are 3cm from table edges, so table is 6cm larger in each direction
    physical_width = marker_width + 6  # 3cm on each side
    physical_height = marker_height + 6  # 3cm on each side
    
    print(f"Marker dimensions: {marker_width}cm x {marker_height}cm")
    print(f"Full table dimensions: {physical_width}cm x {physical_height}cm")
    
    # Define target points for a perfect rectangle (we'll scale this appropriately)
    # Use a scale factor to get reasonable pixel dimensions
    scale_factor = 10  # 10 pixels per cm initially
    target_width = int(physical_width * scale_factor)
    target_height = int(physical_height * scale_factor)
    
    # Calculate the offset to center the markers in the enlarged area
    # Markers should be 3cm (30 pixels at scale_factor=10) from each edge
    offset = 3 * scale_factor
    
    dst_points = np.array([
        [offset, offset],                                    # top_left marker position
        [target_width - offset, offset],                     # top_right marker position  
        [target_width - offset, target_height - offset],     # bottom_right marker position
        [offset, target_height - offset]                     # bottom_left marker position
    ], dtype=np.float32)
    
    # Calculate perspective transform matrix
    transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    
    return transform_matrix, (target_width, target_height)

def apply_perspective_transform(image: np.ndarray, transform_matrix: np.ndarray, output_size: Tuple[int, int]) -> np.ndarray:
    """Apply perspective transform to image"""
    return cv2.warpPerspective(image, transform_matrix, output_size)



def analyze_camera_layout(calibration_data: Dict) -> Dict:
    """
    Analyze camera layout from physical positions in calibration data
    
    Returns:
        dict: Layout information with camera positions
    """
    camera_positions = {}
    
    # Extract center position for each camera from its calibration markers
    for camera_id, camera_data in calibration_data.items():
        markers = camera_data["calibration_markers"]
        
        # Calculate center position from all markers
        x_positions = [marker["physical_position"][0] for marker in markers.values()]
        y_positions = [marker["physical_position"][1] for marker in markers.values()]
        
        center_x = (min(x_positions) + max(x_positions)) / 2
        center_y = (min(y_positions) + max(y_positions)) / 2
        
        camera_positions[camera_id] = (center_x, center_y)
        print(f"Camera {camera_id}: Center at ({center_x:.1f}, {center_y:.1f}) cm")
    
    # Sort cameras into a 2x2 grid based on their physical positions
    # Find x and y thresholds to separate cameras
    x_centers = [pos[0] for pos in camera_positions.values()]
    y_centers = [pos[1] for pos in camera_positions.values()]
    
    x_threshold = np.median(x_centers) if len(x_centers) > 1 else x_centers[0]
    y_threshold = np.median(y_centers) if len(y_centers) > 1 else y_centers[0]
    
    print(f"Layout thresholds: X={x_threshold:.1f}cm, Y={y_threshold:.1f}cm")
    
    # Assign cameras to grid positions
    layout = {
        "top_left": None,     # x < threshold, y < threshold  
        "top_right": None,    # x >= threshold, y < threshold
        "bottom_left": None,  # x < threshold, y >= threshold
        "bottom_right": None  # x >= threshold, y >= threshold
    }
    
    for camera_id, (x, y) in camera_positions.items():
        if x < x_threshold and y < y_threshold:
            layout["top_left"] = camera_id
        elif x >= x_threshold and y < y_threshold:
            layout["top_right"] = camera_id
        elif x < x_threshold and y >= y_threshold:
            layout["bottom_left"] = camera_id
        else:  # x >= x_threshold and y >= y_threshold
            layout["bottom_right"] = camera_id
    
    print("Camera layout:")
    for position, camera_id in layout.items():
        print(f"  {position}: {camera_id if camera_id else 'empty'}")
    
    return layout

def join_images_horizontally(left_image: np.ndarray, right_image: np.ndarray, unified_width: int, unified_height: int) -> np.ndarray:
    """
    Join two images horizontally, handling None/missing images
    
    Args:
        left_image: Left image (or None for empty)
        right_image: Right image (or None for empty) 
        unified_width: Expected width for each image
        unified_height: Expected height for each image
    """
    # Create black placeholder if image is missing
    def get_image_or_placeholder(img):
        if img is None:
            return np.zeros((unified_height, unified_width), dtype=np.uint8)
        return img
    
    left = get_image_or_placeholder(left_image)
    right = get_image_or_placeholder(right_image)
    
    # Ensure both images have the same height
    if left.shape[0] != unified_height:
        left = cv2.resize(left, (unified_width, unified_height))
    if right.shape[0] != unified_height:
        right = cv2.resize(right, (unified_width, unified_height))
    
    # Join horizontally
    return np.hstack([left, right])

def join_images_vertically(top_image: np.ndarray, bottom_image: np.ndarray) -> np.ndarray:
    """Join two images vertically (top above bottom)"""
    if top_image is None and bottom_image is None:
        return None
    
    if top_image is None:
        return bottom_image
    if bottom_image is None:
        return top_image
    
    # Ensure both images have the same width
    h1, w1 = top_image.shape[:2]
    h2, w2 = bottom_image.shape[:2]
    max_width = max(w1, w2)
    
    if w1 != max_width:
        top_image = cv2.resize(top_image, (max_width, h1))
    if w2 != max_width:
        bottom_image = cv2.resize(bottom_image, (max_width, h2))
    
    return np.vstack([top_image, bottom_image])

def create_final_stitched_image(current_frames: Dict, layout: Dict, unified_width: int, unified_height: int) -> np.ndarray:
    """
    Create final stitched image from camera frames using the detected layout
    
    Args:
        current_frames: Dictionary of camera_id -> processed image
        layout: Camera layout from analyze_camera_layout()
        unified_width: Width for each camera image
        unified_height: Height for each camera image
    """
    # Get images for each position (or None if camera not present)
    top_left_img = current_frames.get(layout["top_left"]) if layout["top_left"] else None
    top_right_img = current_frames.get(layout["top_right"]) if layout["top_right"] else None
    bottom_left_img = current_frames.get(layout["bottom_left"]) if layout["bottom_left"] else None
    bottom_right_img = current_frames.get(layout["bottom_right"]) if layout["bottom_right"] else None
    
    # Check if we only have cameras in the top row (1x2 layout)
    has_top_cameras = top_left_img is not None or top_right_img is not None
    has_bottom_cameras = bottom_left_img is not None or bottom_right_img is not None
    
    if has_top_cameras and not has_bottom_cameras:
        # Only top row has cameras - return just the horizontal join (1x2 layout)
        print("Detected 1x2 layout (horizontal only)")
        return join_images_horizontally(top_left_img, top_right_img, unified_width, unified_height)
    
    elif has_bottom_cameras and not has_top_cameras:
        # Only bottom row has cameras - return just the horizontal join (1x2 layout)
        print("Detected 1x2 layout (horizontal only, bottom row)")
        return join_images_horizontally(bottom_left_img, bottom_right_img, unified_width, unified_height)
    
    elif has_top_cameras and has_bottom_cameras:
        # We have cameras in both rows - create full 2x2 grid
        print("Detected 2x2 layout (full grid)")
        top_row = join_images_horizontally(top_left_img, top_right_img, unified_width, unified_height)
        bottom_row = join_images_horizontally(bottom_left_img, bottom_right_img, unified_width, unified_height)
        return join_images_vertically(top_row, bottom_row)
    
    else:
        # No cameras detected - shouldn't happen, but return None
        print("Warning: No cameras detected in layout")
        return None

def setup_camera_transforms():
    """
    Setup camera transforms and parameters once at startup
    
    Returns:
        dict: Setup configuration containing transforms, dimensions, and parameters
    """
    # Load calibration markers
    print("Loading calibration markers...")
    calibration_data = load_calibration_markers("calibration_markers.json")
    
    # === OPTIMIZATION: Calculate transforms ONCE at startup ===
    print("Calculating perspective transforms (one-time setup)...")
    transforms = {}
    output_sizes = {}
    physical_dims = {}
    
    for camera_id in calibration_data.keys():
        print(f"  Setting up camera {camera_id}...")
        transform_matrix, output_size = calculate_perspective_transform(calibration_data[camera_id])
        transforms[camera_id] = transform_matrix
        output_sizes[camera_id] = output_size
        
        # Calculate physical dimensions (add 6cm for the 3cm border on each side)
        markers = calibration_data[camera_id]["calibration_markers"]
        physical_points = [marker["physical_position"] for marker in markers.values()]
        physical_points = np.array(physical_points)
        marker_w = max(physical_points[:, 0]) - min(physical_points[:, 0])
        marker_h = max(physical_points[:, 1]) - min(physical_points[:, 1])
        # Full table is marker area + 6cm (3cm border on each side)
        phys_w = marker_w + 6
        phys_h = marker_h + 6
        physical_dims[camera_id] = (phys_w, phys_h)
        
        print(f"    Transform matrix calculated: {output_size[0]}x{output_size[1]} output")
    
    # Analyze camera layout from physical positions
    print("Analyzing camera layout...")
    layout = analyze_camera_layout(calibration_data)
    
    # Pre-calculate scale normalization parameters
    print("Pre-calculating scale normalization...")
    # Use the first available camera as reference for unified scale
    first_camera_id = next(iter(calibration_data.keys()))
    unified_scale_x = output_sizes[first_camera_id][0] / physical_dims[first_camera_id][0]  # pixels per cm
    unified_scale_y = output_sizes[first_camera_id][1] / physical_dims[first_camera_id][1]  # pixels per cm
    unified_width = int(physical_dims[first_camera_id][0] * unified_scale_x)
    unified_height = int(physical_dims[first_camera_id][1] * unified_scale_y)
    
    print(f"Unified scale: {unified_scale_x:.2f} px/cm (x), {unified_scale_y:.2f} px/cm (y)")
    print(f"Unified dimensions: {unified_width}x{unified_height} pixels per camera")
    print("=== Setup complete! ===")
    
    return {
        "transforms": transforms,
        "output_sizes": output_sizes,
        "physical_dims": physical_dims,
        "layout": layout,
        "unified_width": unified_width,
        "unified_height": unified_height
    }

def process_and_join_streams(setup_config: dict):
    """
    Process camera streams and yield joined output using pre-calculated setup
    
    Args:
        setup_config: Configuration dictionary from setup_camera_transforms()
        
    Yields:
        Stitched images as they are created
    """
    transforms = setup_config["transforms"]
    output_sizes = setup_config["output_sizes"]
    layout = setup_config["layout"]
    unified_width = setup_config["unified_width"]
    unified_height = setup_config["unified_height"]
    
    print("Starting real-time processing...")
    print(f"Expected cameras: {[cam_id for cam_id in layout.values() if cam_id]}")
    
    # Store frames for joining
    frame_buffer = {}
    expected_cameras = set(cam_id for cam_id in layout.values() if cam_id)
    
    frame_count = 0
    for camera_id, frame_data in poll_frame_data():
        # Process the frame
        processed_frame = sharpen_and_rotate_image(buffer_to_array(frame_data))
        
        # Apply perspective transform (fast - just matrix multiplication!)
        if camera_id in transforms:
            transformed = cv2.warpPerspective(
                processed_frame, 
                transforms[camera_id], 
                output_sizes[camera_id]
            )
            
            # Normalize to unified scale (fast resize)
            normalized = cv2.resize(transformed, (unified_width, unified_height))
            
            # Store current frame in buffer
            frame_buffer[camera_id] = normalized
            
            # Show individual camera views
            #cv2.imshow(f"Camera {camera_id} - Original", processed_frame)
            #cv2.imshow(f"Camera {camera_id} - Transformed", normalized)
        
        # Check if we have frames from all expected cameras
        if expected_cameras.issubset(set(frame_buffer.keys())):
            # Create joined image using flexible layout
            final_stitched = create_final_stitched_image(
                frame_buffer, 
                layout, 
                unified_width, 
                unified_height
            )
            
            if final_stitched is not None:
                frame_count += 1
                
                if frame_count % 30 == 0:  # Print status every 30 frames
                    active_cameras = len(frame_buffer)
                    total_expected = len([cam for cam in layout.values() if cam])
                    print(f"Frame {frame_count}: Processing {active_cameras}/{total_expected} cameras at ~30 FPS")
                
                # Yield the stitched image
                yield final_stitched
            
            # Clear buffer after processing
            frame_buffer.clear()
        
        # Minimal key handling for OpenCV windows
        cv2.waitKey(1)

def process_camera_streams():
    """Main function to process camera streams and create joined output"""
    # Setup phase
    setup_config = setup_camera_transforms()
    
    # Processing phase
    for stitched_image in process_and_join_streams(setup_config):
        # The server will handle displaying the stitched image
        # For now, we just yield it
        pass

if __name__ == "__main__":
    process_camera_streams() 