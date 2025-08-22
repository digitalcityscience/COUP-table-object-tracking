#!/usr/bin/env python3
"""
Simple script to run stitching with your 2 camera inputs
"""

import cv2
import numpy as np
import time
import json
import os
from sticthing_v2 import PhysicalMultiCameraStitcher, PhysicalCalibrationConfig, MIN_MARKERS_PER_CAMERA
from mock_camera import poll_frame_data
from detection import detect_markers
from marker import map_detected_markers
from image import sharpen_and_rotate_image

def create_config():
    """Create configuration - adjust these measurements based on your setup"""
    return PhysicalCalibrationConfig(
        # These values are now only used for default calculations
        # Exact marker positions are defined in _calculate_workspace_bounds
        edge_to_marker_x=3.0,
        edge_to_marker_y=3.0,
        marker_spacing_x=74.0,
        marker_spacing_y=74.0,
        total_workspace_width_cm=160.0
    )

def save_calibration_visualization(stitcher, accumulated_marker_data, latest_images):
    """Save visualization of calibration points for each camera and combined view"""
    # Create output directory if it doesn't exist
    os.makedirs("calibration_visualizations", exist_ok=True)
    
    # Create a JSON structure to save calibration points
    calibration_points = {}
    
    # Process each camera
    for camera_id, markers in accumulated_marker_data.items():
        if camera_id not in latest_images or not markers:
            continue
            
        # Get the latest image for this camera
        img = latest_images[camera_id].copy()
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Draw all detected markers
        calibration_points[camera_id] = {}
        for marker_id, marker in markers.items():
            # Extract position
            pos = marker.position if hasattr(marker, 'position') else marker
            x, y = int(pos[0]), int(pos[1])
            
            # Draw marker on image
            cv2.circle(img_color, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(img_color, f"ID:{marker_id}", (x + 10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Save to JSON
            calibration_points[camera_id][str(marker_id)] = {"x": float(pos[0]), "y": float(pos[1])}
            
        # Save individual camera view with markers
        cv2.imwrite(f"calibration_visualizations/camera_{camera_id}_markers.png", img_color)
    
    # Save calibration points as JSON
    with open("calibration_visualizations/calibration_points.json", "w") as f:
        json.dump(calibration_points, f, indent=2)
    
    # Create combined visualization if calibrated
    if stitcher.is_calibrated:
        # Create a color version of the stitched image
        stitched_vis = np.zeros((stitcher.output_size_pixels[1], stitcher.output_size_pixels[0], 3), dtype=np.uint8)
        
        # Draw physical grid lines
        for cm_x in range(0, int(stitcher.workspace_bounds_cm[2]), 10):
            pixel_x = int(cm_x * stitcher.config.pixels_per_cm_x)
            cv2.line(stitched_vis, (pixel_x, 0), (pixel_x, stitched_vis.shape[0]), (0, 50, 0), 1)
        
        for cm_y in range(0, int(stitcher.workspace_bounds_cm[3]), 10):
            pixel_y = int(cm_y * stitcher.config.pixels_per_cm_y)
            cv2.line(stitched_vis, (0, pixel_y), (stitched_vis.shape[1], pixel_y), (0, 50, 0), 1)
        
        # Draw marker positions in physical space
        if hasattr(stitcher, 'marker_positions_cm'):
            for marker_id, (cm_x, cm_y) in stitcher.marker_positions_cm.items():
                pixel_pos = stitcher.physical_to_pixel((cm_x, cm_y))
                cv2.circle(stitched_vis, pixel_pos, 8, (0, 0, 255), -1)
                cv2.putText(stitched_vis, f"ID:{marker_id}", 
                           (pixel_pos[0] + 10, pixel_pos[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Save the combined visualization
        cv2.imwrite("calibration_visualizations/combined_calibration.png", stitched_vis)

def run_stitching():
    """Run the stitching process"""
    print("Starting stitching process...")
    
    # Create stitcher
    config = create_config()
    stitcher = PhysicalMultiCameraStitcher(config)
    
    # Define camera-specific corner marker IDs: Top-left, top-right, bottom-right, bottom-left
    camera_corner_ids = {
        "000": [48, 44, 41, 43],
        "001": [68, 62, 69, 135]
    }
    stitcher.camera_corner_ids = camera_corner_ids
    
    # Accumulated marker data for each camera
    accumulated_marker_data = {}
    calibrated = False
    
    # Store the most recent image for each camera
    latest_images = {}
    
    # Marker collection tracking
    marker_collection_start_time = time.time()
    marker_collection_timeout = 120  # Wait up to 2 minutes for markers
    detected_corner_markers = {}  # Track which corner markers we've seen
    
    try:
        for camera_id, image in poll_frame_data():
            print(f"Processing frame from camera {camera_id}")
            
            # Process image
            ir_image = sharpen_and_rotate_image(image)
            
            # Store the latest image for this camera
            latest_images[camera_id] = ir_image.copy()
            
            # Detect markers
            corners, ids, _ = detect_markers(ir_image)
            building_dict = map_detected_markers(camera_id, ids, corners)
            
            print(f"Camera {camera_id} detected markers: {list(building_dict.keys()) if building_dict else 'None'}")
            
            # Collect calibration data if not calibrated yet
            if not calibrated:
                # Initialize tracking for this camera if not already done
                if camera_id not in detected_corner_markers:
                    detected_corner_markers[camera_id] = set()
                    accumulated_marker_data[camera_id] = {}
                
                # Accumulate marker data over time
                if camera_id in camera_corner_ids:
                    for marker_id in building_dict:
                        # Save all detected markers' positions
                        accumulated_marker_data[camera_id][marker_id] = building_dict[marker_id]
                        
                        # Track corner markers specifically
                        if marker_id in camera_corner_ids[camera_id]:
                            detected_corner_markers[camera_id].add(marker_id)
                
                # Print progress
                detected_counts = {cam: len(markers) for cam, markers in detected_corner_markers.items()}
                expected_counts = {cam: len(camera_corner_ids[cam]) for cam in detected_corner_markers.keys() 
                                if cam in camera_corner_ids}
                
                print(f"Marker collection progress: {detected_counts} of {expected_counts} (need minimum {MIN_MARKERS_PER_CAMERA} per camera)")
                
                # Check if we have enough markers or if we've timed out
                elapsed_time = time.time() - marker_collection_start_time
                
                # Check if we have minimum required markers for each camera
                ready_for_calibration = all(len(markers) >= MIN_MARKERS_PER_CAMERA for cam, markers in detected_corner_markers.items() 
                                         if cam in camera_corner_ids)
                
                # Check if we have all cameras
                all_cameras_present = all(cam in detected_corner_markers for cam in camera_corner_ids)
                
                # Check if we've timed out
                timed_out = elapsed_time > marker_collection_timeout
                
                # Try calibration when ready or timed out
                if (ready_for_calibration and all_cameras_present) or timed_out:
                    print(f"Attempting calibration after {elapsed_time:.1f} seconds")
                    if timed_out:
                        print(f"Warning: Timed out waiting for all markers. Proceeding with available markers.")
                    
                    # Create calibration frames with accumulated marker data
                    calibration_frames = {}
                    for cam_id, markers in accumulated_marker_data.items():
                        if markers and cam_id in latest_images:  # Only include cameras with markers and images
                            calibration_frames[cam_id] = (latest_images[cam_id], markers)
                    
                    # Only attempt calibration if we have data for at least 2 cameras
                    if len(calibration_frames) >= 2:
                        try:
                            stitcher.calibrate_cameras(calibration_frames)
                            stitcher.save_calibration("calibration.json")
                            # Export calibration points to JSON with timestamp
                            stitcher.export_calibration_points_json()
                            calibrated = True
                            print("Calibration successful!")
                            
                            # Save visualization of calibration points
                            save_calibration_visualization(stitcher, accumulated_marker_data, latest_images)
                        except Exception as e:
                            print(f"Calibration failed: {e}")
                            # Reset start time to try again
                            marker_collection_start_time = time.time()
            
            # If calibrated, perform stitching
            if calibrated:
                # Collect current frames for stitching
                if not hasattr(run_stitching, 'current_frames'):
                    run_stitching.current_frames = {}
                
                run_stitching.current_frames[camera_id] = ir_image
                
                # Stitch when we have both cameras
                if len(run_stitching.current_frames) >= 2:
                    stitched = stitcher.stitch_frame(run_stitching.current_frames)
                    
                    if stitched is not None:
                        # Show stitched result
                        cv2.imshow("Stitched View", stitched)
                        
                        # Create a color visualization to better see overlapping areas
                        color_vis = cv2.cvtColor(stitched, cv2.COLOR_GRAY2BGR)
                        
                        # Add borders between camera regions (optional)
                        for cam_id, img in run_stitching.current_frames.items():
                            if cam_id in stitcher.calibrations:
                                # Create a mask for this camera's contribution
                                cal = stitcher.calibrations[cam_id]
                                warped = cv2.warpPerspective(
                                    img, 
                                    cal.homography_matrix,
                                    stitcher.output_size_pixels,
                                    flags=cv2.INTER_LINEAR
                                )
                                mask = warped > 0
                                
                                # Create a border around this camera's region
                                kernel = np.ones((5,5), np.uint8)
                                border = cv2.dilate(mask.astype(np.uint8), kernel) - mask.astype(np.uint8)
                                
                                # Add colored border (different color for each camera)
                                if cam_id == "000":
                                    color_vis[border.astype(bool)] = [0, 0, 255]  # Red for camera 000
                                else:
                                    color_vis[border.astype(bool)] = [0, 255, 0]  # Green for camera 001
                                    
                                # Draw calibration markers in the visualization
                                if cam_id in accumulated_marker_data:
                                    for marker_id, marker in accumulated_marker_data[cam_id].items():
                                        if marker_id in stitcher.camera_corner_ids.get(cam_id, []):
                                            # Get original position
                                            pos = marker.position if hasattr(marker, 'position') else marker
                                            orig_pos = np.array([[pos[0], pos[1]]], dtype=np.float32).reshape(-1, 1, 2)
                                            
                                            # Transform to stitched image coordinates
                                            transformed = cv2.perspectiveTransform(orig_pos, cal.homography_matrix)
                                            x, y = int(transformed[0][0][0]), int(transformed[0][0][1])
                                            
                                            # Draw marker with ID
                                            cv2.circle(color_vis, (x, y), 8, [255, 0, 255], -1)  # Magenta circle
                                            cv2.putText(color_vis, f"ID:{marker_id}", (x + 10, y), 
                                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        cv2.imshow("Overlap Visualization", color_vis)
                        
                        # Save periodically
                        cv2.imwrite("latest_stitched.png", stitched)
                        cv2.imwrite("latest_stitched_vis.png", color_vis)
                    
                    run_stitching.current_frames.clear()
            
            # Show individual camera views
            cv2.imshow(f"Camera {camera_id}", ir_image)
            
            # Break on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_stitching() 