import cv2
import numpy as np
import time
from typing import Dict
import json
import os

from image import buffer_to_array, sharpen_and_rotate_image
from detection import detect_markers, detect_markers_with_refinement
from image import sharpen_and_rotate_image
# from mock_camera import poll_frame_data  # for testing without pyrealsense cameras , using local video file streams instead
from camera import poll_frame_data
from distortion_analysis import analyze_camera_distortion


def save_calibration_markers(camera_setup, timeout: int = 30) -> Dict:
    """
    Find calibration markers in camera streams and save their positions with enhanced visual feedback.
    
    This function now ensures that camera position data is properly preserved in the calibration file
    to prevent coordinate system mismatches and image flipping issues.
    
    Args:
        camera_setup: Dictionary containing camera configurations with position info and exactly 4 calibration markers per camera
            Format: {
                "cam_001": {
                    "position": "top_left",  #
                    "calibration_markers": {
                        "top_left": {"id": "48", "pixel_position": None, "physical_position": [3, 3]},
                        "top_right": {"id": "44", "pixel_position": None, "physical_position": [77, 3]},
                        "bottom_right": {"id": "41", "pixel_position": None, "physical_position": [77, 77]},
                        "bottom_left": {"id": "43", "pixel_position": None, "physical_position": [3, 77]}
                    },
                    "measurements": {"width": 80, "height": 80, "marker_offset": 3}
                },
                ...
            }
        timeout: Maximum time (in seconds) to wait for all markers to be found
    
    Returns:
        Updated camera_setup dictionary with pixel positions filled in and position data preserved
    """
    
    print("\n" + "="*80)
    print("🎯 CALIBRATION MARKER DETECTION - ENHANCED")
    print("="*80)
    print("This process will detect ArUco markers using the coordinate system")
    print("established during the enhanced camera setup phase.")
    print("="*80)
       
    print(f"📷 Starting calibration marker detection with {len(camera_setup)} cameras")
    print(f"⏱️  Will timeout after {timeout} seconds if not all markers are found")
    
    # Validate that camera_setup contains position information
    print(f"\n🔍 VALIDATING CAMERA SETUP:")
    for cam_id, config in camera_setup.items():
        if "position" not in config or config["position"] is None:
            print(f"❌ WARNING: Camera {cam_id} missing position information!")
            print("   This may cause coordinate system issues. Please re-run camera setup.")
        else:
            print(f"✅ Camera {cam_id}: position = {config['position']}")
    
    # Track which markers we still need to find
    markers_to_find = {}
    total_markers = 0
    
    # Initialize tracking structures and validate input
    detected_markers = {}
    best_frames = {}
    
    for camera_id, camera_config in camera_setup.items():
        if "calibration_markers" not in camera_config:
            raise ValueError(f"Camera {camera_id} missing calibration_markers configuration")
        
        markers = camera_config["calibration_markers"]
        required_positions = ["top_left", "top_right", "bottom_right", "bottom_left"]
        
        for pos in required_positions:
            if pos not in markers:
                raise ValueError(f"Camera {camera_id} missing {pos} marker configuration")
            if "id" not in markers[pos]:
                raise ValueError(f"Camera {camera_id} {pos} marker missing ID")
        
        # Track markers we need to find for this camera
        camera_markers = [markers[pos]["id"] for pos in required_positions]
        markers_to_find[camera_id] = camera_markers.copy()
        total_markers += len(camera_markers)
        detected_markers[camera_id] = {}
        
        print(f"📍 Camera {camera_id} ({camera_config.get('position', 'unknown position')}):")
        print(f"   Looking for markers: {camera_markers}")

    print(f"\n🎯 Total markers to detect: {total_markers}")
    print(f"📹 Camera windows will show detected markers in real-time")
    print(f"⌨️  Press 'Q' to quit early (will save partial results)")
    print("="*60)

    start_time = time.time()
    found_markers = 0
    
    print("starting calibration marker detection")
    try:
        # Process camera frames
        for camera_id, image_data in poll_frame_data():
            # Skip cameras not in our config
            if camera_id not in camera_setup:
                print(f"Skipping unknown camera: {camera_id}")
                continue
            
            # Process image
            ir_image = sharpen_and_rotate_image(buffer_to_array(image_data))
            
            # Initialize tracking for this camera if needed
            if camera_id not in detected_markers:
                detected_markers[camera_id] = {}
                best_frames[camera_id] = ir_image.copy()
            
            # Detect markers with refinement
            corners, ids, rejected = detect_markers_with_refinement(ir_image, camera_setup[camera_id])
            
            # Create annotated image with detected markers
            marker_image = ir_image.copy()
            if ids is not None:
                print(f"found markers {ids}")
                marker_image = cv2.aruco.drawDetectedMarkers(marker_image, corners, ids)
                
                # Check each detected marker
                for i, marker_id in enumerate(ids):
                    marker_id_str = str(marker_id[0])
                    
                    # Check if this marker belongs to this camera and hasn't been found yet
                    if marker_id_str in markers_to_find[camera_id]:
                        # Find which position this marker belongs to
                        for position, marker_info in camera_setup[camera_id]["calibration_markers"].items():
                            if marker_info["id"] == marker_id_str:
                                # Extract marker position (center of the marker)
                                marker_corners = corners[i][0]
                                center_x = np.mean(marker_corners[:, 0])
                                center_y = np.mean(marker_corners[:, 1])
                                
                                # Update the pixel position if not already found
                                if "pixel_position" not in marker_info.keys():
                                    camera_setup[camera_id]["calibration_markers"][position]["pixel_position"] = [float(center_x), float(center_y)]
                                    
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
                                    
                                    print(f"✅ Found marker {marker_id} for camera {camera_id} at position {position}: ({center_x:.1f}, {center_y:.1f})")
            
            # Show the camera view with detected markers
            cv2.imshow(f"Camera {camera_id}", marker_image)
            
            # Display progress
            elapsed = time.time() - start_time
            print(f"Progress: {found_markers}/{total_markers} markers found ({elapsed:.1f}s elapsed)")
            
            # Print missing markers
            for cam_id, markers in markers_to_find.items():
                print(f"Missing markers {markers} for cam {cam_id}")
                                 
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
        exit()
    else:
        print(f"Successfully found all {total_markers} calibration markers!")


    print("\n" + "="*80)
    print("💾 SAVING CALIBRATION DATA")
    print("="*80)
    
    # Validate that position data is present before saving
    print("🔍 Final validation before saving:")
    for cam_id, config in camera_setup.items():
        if "position" not in config:
            print(f"❌ ERROR: Camera {cam_id} missing position data!")
            print("   Adding placeholder position data to prevent coordinate system issues.")
            config["position"] = f"unknown_{cam_id}"  # Fallback to prevent errors
        else:
            print(f"✅ Camera {cam_id}: position = {config['position']}")
    
    # Save the results to a file with position data preserved
    with open("calibration_markers.json", "w") as f:
        json.dump(camera_setup, f, indent=2)
    print("✅ Saved calibration data with camera positions to calibration_markers.json")
    
    export_pictures_for_debugging(detected_markers, best_frames)
    
    # Verify the saved file contains position data
    print("\n🔍 Verifying saved calibration file:")
    try:
        with open("calibration_markers.json", "r") as f:
            saved_data = json.load(f)
        
        positions_saved = True
        for cam_id, config in saved_data.items():
            if "position" not in config:
                print(f"❌ ERROR: Position data missing for camera {cam_id} in saved file!")
                positions_saved = False
            else:
                print(f"✅ Camera {cam_id} position saved: {config['position']}")
        
        if positions_saved:
            print("✅ All camera position data successfully saved!")
        else:
            print("❌ WARNING: Some position data missing from saved file!")
            print("   This may cause coordinate system issues during stitching.")
            
    except Exception as e:
        print(f"❌ ERROR: Could not verify saved calibration file: {e}")
       
    # Run distortion analysis on the calibrated markers
    print("\n" + "="*50)
    print("Running distortion analysis...")

    distortion_results = analyze_camera_distortion(camera_setup)
    print("Distortion analysis complete!")
    for cam_id, _config in camera_setup.items():
        if distortion_results[cam_id].get("overall_score") <= 0.8:
            raise ValueError(f"Something went wrong during calibration of camera {cam_id}. Please check  calibration_visualizations/ directory for detailed reports. Try to fix the issue and rerun calibration")

    print("Check the calibration_visualizations/ directory for detailed reports.")



    print("\n" + "="*80)
    print("🎉 CALIBRATION MARKER DETECTION COMPLETE!")
    print("="*80)
    print("✅ All markers detected and saved")
    print("✅ Camera position data preserved")
    print("✅ Coordinate system properly established")
    print("📁 Data saved to: calibration_markers.json")
    print("📁 Visualizations saved to: calibration_visualizations/")
    print("="*80)
    
    return camera_setup


def export_pictures_for_debugging(detected_markers, best_frames):
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
