import cv2
import numpy as np
import time
from typing import Dict, List, Tuple, Optional
import json
import os

def find_calibration_markers(cameras_config: Dict, timeout: int = 6) -> Dict:
    """
    Find calibration markers in camera streams and save their positions.
    
    Args:
        cameras_config: Dictionary containing camera configurations with exactly 4 calibration markers per camera
            Format: {
                "cam_001": {
                    "calibration_markers": {
                        "top_left": {"id": "48", "pixel_position": None, "physical_position": [3, 3]},
                        "top_right": {"id": "44", "pixel_position": None, "physical_position": [77, 3]},
                        "bottom_right": {"id": "41", "pixel_position": None, "physical_position": [77, 77]},
                        "bottom_left": {"id": "43", "pixel_position": None, "physical_position": [3, 77]}
                    }
                },
                ...
            }
        timeout: Maximum time (in seconds) to wait for all markers to be found
    
    Returns:
        Updated cameras_config dictionary with pixel positions filled in
    """
    from detection import detect_markers
    from image import sharpen_and_rotate_image
    from mock_camera import poll_frame_data
    
    print(f"Starting calibration marker detection with {len(cameras_config)} cameras")
    print(f"Will timeout after {timeout} seconds if not all markers are found")
    
    # Track which markers we still need to find
    markers_to_find = {}
    total_markers = 0
    
    # Initialize tracking structures and validate input
    for camera_id, camera_config in cameras_config.items():
        markers_to_find[camera_id] = []
        
        # Check if camera has calibration markers defined
        if "calibration_markers" not in camera_config:
            print(f"Error: Camera {camera_id} has no calibration_markers defined")
            continue
            
        # Verify we have exactly 4 markers
        if len(camera_config["calibration_markers"]) != 4:
            print(f"Warning: Camera {camera_id} should have exactly 4 calibration markers, found {len(camera_config['calibration_markers'])}")
        
        # Add markers to tracking list
        for position, marker_info in camera_config["calibration_markers"].items():
            # Add marker to the list of markers to find
            marker_id = marker_info["id"]
            markers_to_find[camera_id].append(marker_id)
            total_markers += 1
            
            print(f"Camera {camera_id}: Looking for marker {marker_id} at {position}")
    
    # Track found markers
    found_markers = 0
    start_time = time.time()
    
    # Store best frames for each camera
    best_frames = {}
    # Store all detected calibration marker positions
    detected_markers = {}
    
    try:
        # Process camera frames
        for camera_id, image_data in poll_frame_data():
            # Skip cameras not in our config
            if camera_id not in cameras_config:
                print(f"Skipping unknown camera: {camera_id}")
                continue
            
            # Process image
            ir_image = sharpen_and_rotate_image(image_data)
            
            # Initialize tracking for this camera if needed
            if camera_id not in detected_markers:
                detected_markers[camera_id] = {}
                best_frames[camera_id] = ir_image.copy()
            
            # Detect markers
            corners, ids, _ = detect_markers(ir_image)
            
            # Create annotated image with detected markers
            marker_image = ir_image.copy()
            if ids is not None:
                # Draw all detected markers
                marker_image = cv2.aruco.drawDetectedMarkers(marker_image, corners, ids)
                
                # Check for calibration markers
                for i, marker_id in enumerate(ids.flatten()):
                    marker_id_str = str(marker_id)
                    
                    # Check if this is a marker we're looking for
                    if camera_id in markers_to_find and marker_id_str in markers_to_find[camera_id]:
                        # Find which position this marker corresponds to
                        for position, marker_info in cameras_config[camera_id]["calibration_markers"].items():
                            if marker_info["id"] == marker_id_str:
                                # Extract marker position (center of the marker)
                                marker_corners = corners[i][0]
                                center_x = np.mean(marker_corners[:, 0])
                                center_y = np.mean(marker_corners[:, 1])
                                
                                # Update the pixel position if not already found
                                if marker_info["pixel_position"] is None:
                                    cameras_config[camera_id]["calibration_markers"][position]["pixel_position"] = [float(center_x), float(center_y)]
                                    
                                    # Store marker info for final visualization - only center position
                                    detected_markers[camera_id][position] = {
                                        "id": marker_id_str,
                                        "position": [float(center_x), float(center_y)]
                                    }
                                    
                                    # Remove from markers to find
                                    markers_to_find[camera_id].remove(marker_id_str)
                                    found_markers += 1
                                    
                                    # Update best frame if this is the first time we've seen this marker
                                    best_frames[camera_id] = ir_image.copy()
                                    
                                    print(f"Found marker {marker_id} for camera {camera_id} at position {position}: ({center_x:.1f}, {center_y:.1f})")
            
            # Show the camera view with detected markers
            cv2.imshow(f"Camera {camera_id}", marker_image)
            
            # Display progress
            elapsed = time.time() - start_time
            print(f"Progress: {found_markers}/{total_markers} markers found ({elapsed:.1f}s elapsed)")
            
            # Check if we've found all markers for this camera
            camera_complete = []
            for cam_id, markers in markers_to_find.items():
                if len(markers) == 0:
                    camera_complete.append(cam_id)
            
            if camera_complete:
                print(f"Completed cameras: {camera_complete}")
            
            # Check if we've found all markers
            remaining_markers = sum(len(markers) for markers in markers_to_find.values())
            if remaining_markers == 0:
                print("All calibration markers found!")
                break
            
            # Check if we've timed out
            if elapsed > timeout:
                print(f"Timeout after {elapsed:.1f}s. Found {found_markers}/{total_markers} markers.")
                break
            
            # Break on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Manually stopped marker detection")
                break
    
    except KeyboardInterrupt:
        print("Marker detection interrupted")
    finally:
        cv2.destroyAllWindows()
    
    # Report final status
    if found_markers < total_markers:
        missing_markers = {}
        for camera_id, markers in markers_to_find.items():
            if markers:
                missing_markers[camera_id] = markers
        
        print(f"Warning: Not all markers were found. Missing: {missing_markers}")
    else:
        print(f"Successfully found all {total_markers} calibration markers!")
    
    # Save the results to a file
    with open("calibration_markers.json", "w") as f:
        json.dump(cameras_config, f, indent=2)
    print("Saved marker positions to calibration_markers.json")
    
    # Save calibration points to a separate file for visualization
    os.makedirs("calibration_visualizations", exist_ok=True)
    with open("calibration_visualizations/calibration_points.json", "w") as f:
        json.dump(detected_markers, f, indent=2)
    
    # Create and save final annotated images with ALL detected calibration markers
    final_annotated_images = {}
    for camera_id, frame in best_frames.items():
        # Create a color version of the frame for better annotations
        color_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        # Draw all detected markers for this camera
        if camera_id in detected_markers:
            for position, marker_info in detected_markers[camera_id].items():
                # Draw marker position
                center_x, center_y = marker_info["position"]
                marker_id = marker_info["id"]
                
                # Draw a colored circle at marker center
                if position == "top_left":
                    color = (0, 0, 255)  # Red
                elif position == "top_right":
                    color = (0, 255, 0)  # Green
                elif position == "bottom_right":
                    color = (255, 0, 0)  # Blue
                elif position == "bottom_left":
                    color = (255, 255, 0)  # Cyan
                else:
                    color = (255, 0, 255)  # Magenta
                
                # Draw circle and label
                cv2.circle(color_frame, (int(center_x), int(center_y)), 10, color, -1)
                cv2.putText(color_frame, f"{position} (ID:{marker_id})", 
                           (int(center_x) + 15, int(center_y)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        final_annotated_images[camera_id] = color_frame
    
    # Save the final annotated images
    for camera_id, image in final_annotated_images.items():
        output_path = f"calibration_visualizations/camera_{camera_id}_markers.png"
        cv2.imwrite(output_path, image)
        print(f"Saved annotated marker visualization for camera {camera_id} to {output_path}")
    
    # Create a combined visualization if we have multiple cameras
    if len(final_annotated_images) > 1:
        # Get dimensions
        heights = [img.shape[0] for img in final_annotated_images.values()]
        widths = [img.shape[1] for img in final_annotated_images.values()]
        max_height = max(heights)
        total_width = sum(widths)
        
        # Create combined image
        combined = np.zeros((max_height, total_width, 3), dtype=np.uint8)
        x_offset = 0
        for camera_id, img in final_annotated_images.items():
            h, w = img.shape[:2]
            combined[0:h, x_offset:x_offset+w] = img
            
            # Add camera label
            cv2.putText(combined, f"Camera {camera_id}", 
                       (x_offset + 10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
            
            x_offset += w
        
        # Save combined visualization
        cv2.imwrite("calibration_visualizations/combined_calibration.png", combined)
        print("Saved combined calibration visualization")
    
    return cameras_config

# Example usage
if __name__ == "__main__":
    # Example configuration with exactly 4 markers per camera
    cameras = {
        "000": {
            "calibration_markers": {
                "top_left": {"id": "48", "pixel_position": None, "physical_position": [3, 3]},
                "top_right": {"id": "44", "pixel_position": None, "physical_position": [77, 3]},
                "bottom_right": {"id": "41", "pixel_position": None, "physical_position": [77, 77]},
                "bottom_left": {"id": "43", "pixel_position": None, "physical_position": [3, 77]}
            }
        },
        "001": {
            "calibration_markers": {
                "top_left": {"id": "68", "pixel_position": None, "physical_position": [83, 3]},
                "top_right": {"id": "62", "pixel_position": None, "physical_position": [157, 3]},
                "bottom_right": {"id": "69", "pixel_position": None, "physical_position": [157, 77]},
                "bottom_left": {"id": "999", "pixel_position": [357, 762], "physical_position": [83, 77]}
            }
        }
    }
    find_calibration_markers(cameras)