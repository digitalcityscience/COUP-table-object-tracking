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

def normalize_scale(images: Dict[str, np.ndarray], physical_dims: Dict[str, Tuple[float, float]]) -> Dict[str, np.ndarray]:
    """
    Normalize images to have the same scale (pixels per cm)
    
    Args:
        images: Dictionary of camera_id -> transformed image
        physical_dims: Dictionary of camera_id -> (width_cm, height_cm)
    
    Returns:
        Dictionary of camera_id -> rescaled image
    """
    # Calculate pixels per cm for each camera
    scales = {}
    for camera_id, image in images.items():
        h, w = image.shape[:2]
        phys_w, phys_h = physical_dims[camera_id]
        scale_x = w / phys_w
        scale_y = h / phys_h
        scales[camera_id] = (scale_x, scale_y)
        print(f"Camera {camera_id}: {scale_x:.2f} px/cm (x), {scale_y:.2f} px/cm (y)")
    
    # Find the minimum scale to ensure all images fit
    min_scale_x = min(scale[0] for scale in scales.values())
    min_scale_y = min(scale[1] for scale in scales.values())
    
    print(f"Using unified scale: {min_scale_x:.2f} px/cm (x), {min_scale_y:.2f} px/cm (y)")
    
    # Rescale all images to the same scale
    normalized_images = {}
    for camera_id, image in images.items():
        phys_w, phys_h = physical_dims[camera_id]
        new_width = int(phys_w * min_scale_x)
        new_height = int(phys_h * min_scale_y)
        
        normalized_images[camera_id] = cv2.resize(image, (new_width, new_height))
        print(f"Camera {camera_id}: resized to {new_width}x{new_height}")
    
    return normalized_images

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

def process_camera_streams():
    """Main function to process camera streams and create joined output"""
    
    # Load calibration markers
    print("Loading calibration markers...")
    calibration_data = load_calibration_markers("calibration_markers.json")
    
    # Calculate perspective transforms for both cameras
    transforms = {}
    output_sizes = {}
    physical_dims = {}
    
    for camera_id in ["000", "001"]:
        print(f"\nProcessing camera {camera_id}...")
        transform_matrix, output_size = calculate_perspective_transform(calibration_data[camera_id])
        transforms[camera_id] = transform_matrix
        output_sizes[camera_id] = output_size
        
        # Calculate physical dimensions
        markers = calibration_data[camera_id]["calibration_markers"]
        physical_points = [marker["physical_position"] for marker in markers.values()]
        physical_points = np.array(physical_points)
        phys_w = max(physical_points[:, 0]) - min(physical_points[:, 0])
        phys_h = max(physical_points[:, 1]) - min(physical_points[:, 1])
        physical_dims[camera_id] = (phys_w, phys_h)
    
    print("\nStarting camera stream processing...")
    print("Press 'q' to quit, 's' to save current frame")
    
    try:
        frame_count = 0
        for camera_id, frame_data in poll_frame_data():
            # Process the frame
            processed_frame = sharpen_and_rotate_image(frame_data)
            
            # Apply perspective transform
            if camera_id in transforms:
                transformed = apply_perspective_transform(
                    processed_frame, 
                    transforms[camera_id], 
                    output_sizes[camera_id]
                )
                
                # Save individual transformed frame
                cv2.imwrite(f"transformed_{camera_id}.png", transformed)
                
                # Show individual camera view
                cv2.imshow(f"Camera {camera_id} - Original", processed_frame)
                cv2.imshow(f"Camera {camera_id} - Transformed", transformed)
            
            frame_count += 1
            
            # Every few frames, create joined image if we have both cameras
            if frame_count % 2 == 0:  # Process every other frame to get both cameras
                try:
                    # Load the two most recent transformed images
                    img_000 = cv2.imread("transformed_000.png", cv2.IMREAD_GRAYSCALE)
                    img_001 = cv2.imread("transformed_001.png", cv2.IMREAD_GRAYSCALE)
                    
                    if img_000 is not None and img_001 is not None:
                        # Normalize scales
                        images = {"000": img_000, "001": img_001}
                        normalized_images = normalize_scale(images, physical_dims)
                        
                        # Join images
                        joined = join_images_horizontally(
                            normalized_images["000"], 
                            normalized_images["001"]
                        )
                        
                        # Save and show joined result
                        cv2.imwrite("latest_joined.png", joined)
                        cv2.imshow("Joined View", joined)
                        
                        print(f"Frame {frame_count}: Updated joined view")
                
                except Exception as e:
                    print(f"Error creating joined image: {e}")
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('s'):
                timestamp = int(time.time())
                # Save current state with timestamp
                try:
                    img_000 = cv2.imread("transformed_000.png", cv2.IMREAD_GRAYSCALE)
                    img_001 = cv2.imread("transformed_001.png", cv2.IMREAD_GRAYSCALE)
                    joined = cv2.imread("latest_joined.png", cv2.IMREAD_GRAYSCALE)
                    
                    if img_000 is not None:
                        cv2.imwrite(f"saved_camera_000_{timestamp}.png", img_000)
                    if img_001 is not None:
                        cv2.imwrite(f"saved_camera_001_{timestamp}.png", img_001)
                    if joined is not None:
                        cv2.imwrite(f"saved_joined_{timestamp}.png", joined)
                    
                    print(f"Saved current frame set with timestamp {timestamp}")
                except Exception as e:
                    print(f"Error saving images: {e}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    process_camera_streams() 