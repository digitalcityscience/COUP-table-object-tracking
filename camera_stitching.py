import cv2
import numpy as np
import json
import time
from typing import Dict, Tuple
from mock_camera import poll_frame_data
from image import sharpen_and_rotate_image

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



def join_images_horizontally(image_000: np.ndarray, image_001: np.ndarray) -> np.ndarray:
    """Join two images horizontally (000 on left, 001 on right)"""
    h1, w1 = image_000.shape[:2]
    h2, w2 = image_001.shape[:2]
    
    # Make sure both images have the same height
    max_height = max(h1, h2)
    
    if h1 != max_height:
        image_000 = cv2.resize(image_000, (w1, max_height))
    if h2 != max_height:
        image_001 = cv2.resize(image_001, (w2, max_height))
    
    # Join horizontally
    joined = np.hstack([image_000, image_001])
    
    return joined

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
    
    # Pre-calculate scale normalization parameters
    print("Pre-calculating scale normalization...")
    # Since both tables are the same size, we can use a fixed scale
    unified_scale_x = output_sizes["000"][0] / physical_dims["000"][0]  # pixels per cm
    unified_scale_y = output_sizes["000"][1] / physical_dims["000"][1]  # pixels per cm
    unified_width = int(physical_dims["000"][0] * unified_scale_x)
    unified_height = int(physical_dims["000"][1] * unified_scale_y)
    
    print(f"Unified scale: {unified_scale_x:.2f} px/cm (x), {unified_scale_y:.2f} px/cm (y)")
    print(f"Unified dimensions: {unified_width}x{unified_height} pixels")
    print("=== Setup complete! ===")
    
    return {
        "transforms": transforms,
        "output_sizes": output_sizes,
        "physical_dims": physical_dims,
        "unified_width": unified_width,
        "unified_height": unified_height
    }

def process_and_join_streams(setup_config: dict):
    """
    Process camera streams and create joined output using pre-calculated setup
    
    Args:
        setup_config: Configuration dictionary from setup_camera_transforms()
    """
    transforms = setup_config["transforms"]
    output_sizes = setup_config["output_sizes"]
    unified_width = setup_config["unified_width"]
    unified_height = setup_config["unified_height"]
    
    print("Starting real-time processing...")
    print("Press Ctrl+C to quit")
    
    # Store frames for joining
    current_frames = {}
    
    try:
        frame_count = 0
        for camera_id, frame_data in poll_frame_data():
            # Process the frame
            processed_frame = sharpen_and_rotate_image(frame_data)
            
            # Apply perspective transform (fast - just matrix multiplication!)
            if camera_id in transforms:
                transformed = cv2.warpPerspective(
                    processed_frame, 
                    transforms[camera_id], 
                    output_sizes[camera_id]
                )
                
                # Normalize to unified scale (fast resize)
                normalized = cv2.resize(transformed, (unified_width, unified_height))
                
                # Store current frame
                current_frames[camera_id] = normalized
                
                # Show individual camera views
                cv2.imshow(f"Camera {camera_id} - Original", processed_frame)
                cv2.imshow(f"Camera {camera_id} - Transformed", normalized)
            
            frame_count += 1
            
            # Create joined image when we have both cameras
            if len(current_frames) == 2 and "000" in current_frames and "001" in current_frames:
                # Join images (camera 000 on left, 001 on right)
                joined = join_images_horizontally(current_frames["000"], current_frames["001"])
                
                # Show joined result
                cv2.imshow("Joined View", joined)
                
                if frame_count % 30 == 0:  # Print status every 30 frames
                    print(f"Frame {frame_count}: Processing at ~30 FPS")
            
            # Minimal key handling for OpenCV windows
            cv2.waitKey(1)
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        cv2.destroyAllWindows()

def process_camera_streams():
    """Main function to process camera streams and create joined output"""
    # Setup phase
    setup_config = setup_camera_transforms()
    
    # Processing phase
    process_and_join_streams(setup_config)

if __name__ == "__main__":
    process_camera_streams() 