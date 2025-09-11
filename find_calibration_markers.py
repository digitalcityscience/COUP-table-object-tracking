import cv2
import numpy as np
import time
from typing import Dict, List, Tuple, Optional
import json
import os
import math
from image import buffer_to_array, sharpen_and_rotate_image
from hud import draw_monitor_window, draw_status_window
from detection import detect_markers
from marker import Markers, map_detected_markers

def check_existing_calibration(file_path: str = "calibration_markers.json") -> bool:
    """
    Check if calibration file exists with valid pixel positions
    
    Returns:
        True if calibration exists and is complete, False otherwise
    """
    if not os.path.exists(file_path):
        print(f"No existing calibration file found at {file_path}")
        return False
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Check if all cameras have pixel positions
        for camera_id, camera_data in data.items():
            if "calibration_markers" not in camera_data:
                return False
            
            for position, marker_info in camera_data["calibration_markers"].items():
                if marker_info.get("pixel_position") is None:
                    print(f"Camera {camera_id} marker at {position} missing pixel position")
                    return False
        
        print(f"Found complete calibration for {len(data)} cameras")
        return True
        
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error reading calibration file: {e}")
        return False



def find_calibration_markers(cameras_config: Dict, timeout: int = 270) -> Dict:
    """
    Find calibration markers in camera streams and save their positions.
    
    First checks if calibration already exists and verifies markers are still visible.
    If existing calibration is found but no markers are visible, aborts.
    
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
    # Check if calibration already exists
    if check_existing_calibration():
        print("Found existing calibration file with complete pixel positions.")
        
        # Load existing calibration
        with open("calibration_markers.json", 'r') as f:
            existing_config = json.load(f)
        
        print("✓ Using existing calibration - calibration markers no longer need to be visible")
        return existing_config
    
    print("No existing calibration found. Starting fresh calibration...")
    from detection import detect_markers
    from image import sharpen_and_rotate_image
    # from mock_camera import poll_frame_data  # for testing without pyrealsense cameras , using local video file streams instead
    from camera import poll_frame_data
    
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
            ir_image = sharpen_and_rotate_image(buffer_to_array(image_data))
            
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
    
       
    # Run distortion analysis on the calibrated markers
    print("\n" + "="*50)
    print("Running distortion analysis...")
    try:
        from distortion_analysis import analyze_camera_distortion
        distortion_results = analyze_camera_distortion(cameras_config)
        print("Distortion analysis complete!")
        print("Check the calibration_visualizations/ directory for detailed reports.")
    except ImportError:
        print("Warning: Could not import distortion_analysis module. Skipping distortion analysis.")
    except Exception as e:
        print(f"Warning: Distortion analysis failed: {e}")
        print("Continuing without distortion analysis...")
    
    return cameras_config

def get_available_camera_ids():
    """
    Try to get camera IDs from device manager, fallback to manual input
    """
    try:
        # Try to import and use real device manager
        from realsense.realsense_device_manager import DeviceManager
        import pyrealsense2 as rs
        
        # Setup device manager
        c = rs.config()
        c.enable_stream(rs.stream.infrared, 1, 1280, 720, rs.format.y8, 6)
        device_manager = DeviceManager(rs.context(), c)
        device_manager.enable_all_devices()
        
        # Try to get connected device IDs
        available_ids = device_manager.get_enabled_devices_ids()
        if available_ids:
            print(f"Found {len(available_ids)} cameras: {available_ids}")
            return [id[-3:] for id in available_ids]
        else:
            print("No cameras detected via device manager")
            raise Exception("No cameras detected")

            
    except (ImportError, Exception) as e:
        print(f"Could not access device manager: {e}")
        print("Make sure the cameras are connected and powered on")
        raise Exception("No cameras detected")

def show_camera_streams_for_identification(cam_to_show):
    """
    Show camera streams to help user identify which camera is which
    """
    print("Showing camera streams for identification...")
    print("Press 'q' to stop viewing and continue with setup")
    
    try:
        # from mock_camera import poll_frame_data   # for use without physical cameras
        from camera import poll_frame_data
        from image import sharpen_and_rotate_image
        
        # Show streams for a few seconds to let user see each camera
        for camera_id, frame_data in poll_frame_data():
            if camera_id != cam_to_show:
                continue
            print( 'test',camera_id,frame_data)

           # processed_frame = frame_data
            processed_frame = sharpen_and_rotate_image(buffer_to_array(frame_data))
            # print("act",active_cameras)

            #corners, ids, rejectedImgPoints = detect_markers(processed_frame)
            #buildingDict = map_detected_markers(camera_id, ids, corners)
            #draw_monitor_window(processed_frame, corners, rejectedImgPoints, camera_id)
            # draw_status_window(buildingDict, camera_id)

            
            # Show the camera view
            #cv2.imshow(f"Camera {camera_id} - Enter calibration marker ids for this camera", processed_frame)
            # time.sleep(4)
            break
        
        corners, ids, rejectedImgPoints = detect_markers(processed_frame)
        buildingDict = map_detected_markers(camera_id, ids, corners)
        draw_monitor_window(processed_frame, corners, rejectedImgPoints, camera_id)
        draw_status_window(buildingDict, camera_id)
        #cv2.imshow(f"Camera {camera_id} - Enter calibration marker ids for this camera", processed_frame)
        #time.sleep(4)
        # cv2.destroyAllWindows()
        
        
    except Exception as e:
        print(f"Could not show camera streams: {e}")
        return None

def prompt_calibration_setup() -> Dict:
    """
    Prompt for basic calibration data and create cameras config
    """
    print("No existing calibration found. Setting up new calibration...")
    
    # Get table size
    table_width = float(input("Table width (cm): "))
    table_height = float(input("Table height (cm): "))
    
    # Get marker offset
    marker_offset = float(input("Distance from marker to table edge (cm): "))
    
    # Try to get camera IDs automatically
    camera_ids = ["104", "863"]
    
    # Get marker IDs for each camera
    cameras_config = {}
    
    for camera_id in camera_ids:
        # Show camera streams to help identify
        # show_camera_streams_for_identification(camera_id)
        
        print(f"Marker IDs for camera {camera_id}:")
        marker_ids = {}
        for pos in ["top_left", "top_right", "bottom_right", "bottom_left"]:
            marker_id = input(f"  {pos} marker ID: ").strip()
            if not marker_id:
                raise ValueError(f"Marker ID for {pos} cannot be empty")
            marker_ids[pos] = marker_id
        
        cameras_config[camera_id] = {
            "calibration_markers": {
                "top_left": {
                    "id": marker_ids["top_left"],
                    "pixel_position": None,
                    "physical_position": [marker_offset, marker_offset]
                },
                "top_right": {
                    "id": marker_ids["top_right"],
                    "pixel_position": None,
                    "physical_position": [table_width - marker_offset, marker_offset]
                },
                "bottom_right": {
                    "id": marker_ids["bottom_right"],
                    "pixel_position": None,
                    "physical_position": [table_width - marker_offset, table_height - marker_offset]
                },
                "bottom_left": {
                    "id": marker_ids["bottom_left"],
                    "pixel_position": None,
                    "physical_position": [marker_offset, table_height - marker_offset]
                }
            }
        }
    
    return cameras_config

# Example usage
if __name__ == "__main__":
    # Check if calibration already exists
    if check_existing_calibration():
        print("Using existing calibration file...")
        with open("calibration_markers.json", 'r') as f:
            cameras = json.load(f)
    else:
        # No existing calibration - prompt user for configuration
        cameras = prompt_calibration_setup()
    
    # Run calibration marker detection
    find_calibration_markers(cameras)