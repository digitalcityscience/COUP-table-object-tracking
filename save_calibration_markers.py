import cv2
import numpy as np
import time
from typing import Dict
import json
import os

from image import buffer_to_array, sharpen_and_rotate_image
from detection import detect_markers
from detection import detect_markers
from image import sharpen_and_rotate_image
# from mock_camera import poll_frame_data  # for testing without pyrealsense cameras , using local video file streams instead
from camera import poll_frame_data
from distortion_analysis import analyze_camera_distortion



REQUIRED_POSITIONS = ["top_left", "top_right", "bottom_right", "bottom_left"]


def _prompt_marker_id(camera_id: str, position: str) -> str:
    """Ask the user for the ArUco marker ID belonging to a given camera/position."""
    label = position.replace('_', '-').upper()
    while True:
        marker_id = input(f"  📍 Camera {camera_id} - {label} corner marker ID: ").strip()
        if not marker_id:
            print("     ❌ Please enter a marker ID.")
            continue
        try:
            int(marker_id)
            return marker_id
        except ValueError:
            print("     ❌ Please enter a numeric marker ID.")


def save_calibration_markers(camera_setup, timeout: int = 60) -> Dict:
    """
    Interactively find calibration markers in camera streams and save their positions.

    For each camera, and for each of its 4 calibration marker positions (top_left,
    top_right, bottom_right, bottom_left), the user is asked to enter the ArUco
    marker ID that belongs to that position. The live camera stream is then watched
    until that specific marker is detected before moving on to the next position.
    This avoids having to search for every marker on every camera at once and gives
    immediate feedback per marker.

    This function also ensures that camera position data is properly preserved in the
    calibration file to prevent coordinate system mismatches and image flipping issues.

    Args:
        camera_setup: Dictionary containing camera configurations with position info and exactly 4 calibration marker positions per camera
            Format: {
                "cam_001": {
                    "position": "top_left",  #
                    "calibration_markers": {
                        "top_left": {"physical_position": [3, 3]},
                        "top_right": {"physical_position": [77, 3]},
                        "bottom_right": {"physical_position": [77, 77]},
                        "bottom_left": {"physical_position": [3, 77]}
                    },
                    "measurements": {"width": 80, "height": 80, "marker_offset": 3}
                },
                ...
            }
        timeout: Maximum time (in seconds) to wait for each individual marker to be found
    
    Returns:
        Updated camera_setup dictionary with marker IDs, pixel positions filled in and position data preserved
    """
    
    print("\n" + "="*80)
    print("🎯 CALIBRATION MARKER DETECTION - INTERACTIVE")
    print("="*80)
    print("For each camera and marker position, you'll be asked for the marker's")
    print("ArUco ID. The camera feed is then watched until that specific marker")
    print("is found before moving on to the next one.")
    print("="*80)
       
    print(f"📷 Starting calibration marker detection with {len(camera_setup)} cameras")
    print(f"⏱️  Will timeout after {timeout} seconds per marker if it isn't found")
    
    # Validate that camera_setup contains position information and marker structure
    print(f"\n🔍 VALIDATING CAMERA SETUP:")
    for cam_id, config in camera_setup.items():
        if "position" not in config or config["position"] is None:
            print(f"❌ WARNING: Camera {cam_id} missing position information!")
            print("   This may cause coordinate system issues. Please re-run camera setup.")
        else:
            print(f"✅ Camera {cam_id}: position = {config['position']}")

        if "calibration_markers" not in config:
            raise ValueError(f"Camera {cam_id} missing calibration_markers configuration")

        for pos in REQUIRED_POSITIONS:
            if pos not in config["calibration_markers"]:
                raise ValueError(f"Camera {cam_id} missing {pos} marker configuration")

    # Initialize tracking structures
    detected_markers = {camera_id: {} for camera_id in camera_setup}
    best_frames = {}

    print(f"\n📹 Camera windows will show detected markers in real-time")
    print(f"⌨️  Press 'Q' at any time to abort (partial results will be lost)")
    print("="*60)

    frame_iterator = poll_frame_data()

    try:
        for camera_id, camera_config in camera_setup.items():
            print(f"\n" + "-"*60)
            print(f"📍 Camera {camera_id} ({camera_config.get('position', 'unknown position')})")
            print("-"*60)

            for position in REQUIRED_POSITIONS:
                marker_id = _prompt_marker_id(camera_id, position)
                camera_config["calibration_markers"][position]["id"] = marker_id

                print(f"🔎 Searching for marker {marker_id} on camera {camera_id} ({position})...")
                start_time = time.time()
                found = False

                while not found:
                    frame_camera_id, image_data = next(frame_iterator)

                    # Only process frames for the camera we're currently working on
                    if frame_camera_id != camera_id:
                        continue

                    # Process image
                    ir_image = sharpen_and_rotate_image(buffer_to_array(image_data))
                    best_frames[camera_id] = ir_image.copy()

                    # Detect markers
                    corners, ids, _ = detect_markers(ir_image)

                    # Create annotated image with detected markers
                    marker_image = ir_image.copy()
                    if ids is not None:
                        marker_image = cv2.aruco.drawDetectedMarkers(marker_image, corners, ids)

                        for i, detected_id in enumerate(ids):
                            if str(int(detected_id)) == marker_id:
                                # Extract marker position (center of the marker)
                                marker_corners = corners[i][0]
                                center_x = float(np.mean(marker_corners[:, 0]))
                                center_y = float(np.mean(marker_corners[:, 1]))

                                camera_config["calibration_markers"][position]["pixel_position"] = [center_x, center_y]
                                detected_markers[camera_id][position] = {
                                    "id": marker_id,
                                    "position": [center_x, center_y]
                                }

                                found = True
                                print(f"✅ Found marker {marker_id} for camera {camera_id} at position {position}: ({center_x:.1f}, {center_y:.1f})")
                                break

                    # Show the camera view with detected markers
                    cv2.imshow(f"Camera {camera_id}", marker_image)

                    elapsed = time.time() - start_time
                    if not found and elapsed > timeout:
                        print(f"⏱️  Timeout after {elapsed:.1f}s while looking for marker {marker_id}.")
                        retry = input("     Enter 'r' to re-enter the marker ID, or press Enter to keep waiting: ").strip().lower()
                        if retry == 'r':
                            marker_id = _prompt_marker_id(camera_id, position)
                            camera_config["calibration_markers"][position]["id"] = marker_id
                            print(f"🔎 Searching for marker {marker_id} on camera {camera_id} ({position})...")
                        start_time = time.time()

                    # Abort on 'q' key
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("Manually stopped marker detection")
                        raise KeyboardInterrupt

    except KeyboardInterrupt:
        print("Marker detection interrupted")
    finally:
        cv2.destroyAllWindows()

    # Report final status
    total_markers = len(camera_setup) * len(REQUIRED_POSITIONS)
    found_markers = sum(len(positions) for positions in detected_markers.values())

    if found_markers < total_markers:
        missing_markers = {}
        for camera_id in camera_setup:
            missing = [pos for pos in REQUIRED_POSITIONS if pos not in detected_markers.get(camera_id, {})]
            if missing:
                missing_markers[camera_id] = missing
        
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


def export_pictures_for_debugging(
    detected_markers, best_frames, output_dir="calibration_visualizations"
):
    # Save calibration points to a separate file for visualization
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "calibration_points.json"), "w") as f:
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
        output_path = os.path.join(output_dir, f"camera_{camera_id}_markers.png")
        cv2.imwrite(output_path, image)
        print(f"Saved annotated marker visualization for camera {camera_id} to {output_path}")
