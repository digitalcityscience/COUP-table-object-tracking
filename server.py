import asyncio
import socket
import signal
import sys

from marker import Markers, map_detected_markers
from tracker import track_v2
from time import time_ns
from detection import detect_markers
from hud import draw_monitor_window, draw_status_window
from find_calibration_markers import find_calibration_markers
from camera_stitching import setup_camera_transforms, process_and_join_streams
import cv2

from collections import defaultdict
from typing import Dict, List
import os
from datetime import datetime

# Global variable for stitching setup
stitching_setup = None

socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
SERVER_SETTINGS = ("localhost", 8052)
print(f"Listening to socket connections on: {SERVER_SETTINGS}")
socket.bind(SERVER_SETTINGS)
socket.listen(1)
socket.setblocking(False)
loop = asyncio.new_event_loop()


def initialize_camera_stitching():
    """Initialize camera stitching setup"""
    global stitching_setup
    
    print("=== Initializing Camera Stitching System ===")
    
    # Define calibration markers configuration 
    cameras_config = {
        "104": {
            "calibration_markers": {
                "top_left": {"id": "45", "pixel_position": None, "physical_position": [3, 3]},
                "top_right": {"id": "46", "pixel_position": None, "physical_position": [77, 3]},
                "bottom_right": {"id": "47", "pixel_position": None, "physical_position": [77, 77]},
                "bottom_left": {"id": "44", "pixel_position": None, "physical_position": [3, 77]}
            }
        },
        "863": {
            "calibration_markers": {
                "top_left": {"id": "41", "pixel_position": None, "physical_position": [83, 3]},
                "top_right": {"id": "42", "pixel_position": None, "physical_position": [157, 3]},
                "bottom_right": {"id": "43", "pixel_position": None, "physical_position": [157, 77]},
                "bottom_left": {"id": "40", "pixel_position": None, "physical_position": [83, 77]}
            }
        }
    }
    
    try:
        # Step 1: Run calibration setup
        print("Step 1: Running calibration setup...")
        calibrated_config = find_calibration_markers(cameras_config)
        
        # Step 2: Setup camera transforms
        print("Step 2: Setting up camera transforms...")
        stitching_setup = setup_camera_transforms()
        
        # Step 3: Export a sample stitched image
        print("Step 3: Exporting stitched sample...")
        try:
            # Get one stitched frame as a sample
            for stitched_image in process_and_join_streams(stitching_setup):
                # Export the first stitched result as sample
                os.makedirs("calibration_visualizations", exist_ok=True)
                sample_path = "calibration_visualizations/stitched_sample.png"
                cv2.imwrite(sample_path, stitched_image)
                print(f"✓ Exported stitched sample to {sample_path}")
                break  # Only export one sample, then continue
        except Exception as e:
            print(f"Warning: Could not export stitched sample: {e}")
        
        print("✓ Camera stitching system initialized successfully!")
        return stitching_setup
        
    except Exception as e:
        print(f"✗ Failed to initialize camera stitching: {e}")
        raise


async def main():
    # Initialize camera stitching system at startup
    global stitching_setup
    stitching_setup = initialize_camera_stitching()
    
    while True:
        connection, client_address = await loop.sock_accept(socket)
        print(f"Connection from: {client_address}")
        loop.create_task(send_tracking_matches(connection))


async def send_tracking_matches(connection):
    global stitching_setup
    markers_holder = Markers()
    last_sent = time_ns()
    
    # Iterate over stitched images from process_and_join_streams
    for stitched_image in process_and_join_streams(stitching_setup):
        # Run marker detection on stitched image
        corners, ids, rejectedImgPoints = detect_markers(stitched_image) # runs detection.
        buildingDict = map_detected_markers("stitched", ids, corners)
        
        # Show stitched result with markers
        draw_monitor_window(stitched_image, corners, rejectedImgPoints, "stitched")
        draw_status_window(buildingDict, "stitched")
        
        # Create pseudo-frame for tracking
        stitched_frame = ("stitched", stitched_image)
        markers_holder.addMarkers(track_v2(stitched_frame))  # TODO detection is run twice. already in dobos code.
        
        # Send data to Unity client
        if (time_ns() - last_sent > 200_000_000):
            markers_json = markers_holder.toJSON()
            print("Sending to unity:", markers_json)
            last_sent = time_ns()
            markers_holder.clear()
            await loop.sock_sendall(connection, markers_json.encode("utf-8"))


def shutdown_handler(sig, frame):
    """Handle graceful shutdown"""
    print("\nShutting down server...")
    print("Closing socket...")
    socket.close()
    print("Stopping event loop...")
    loop.stop()
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, shutdown_handler)  # Ctrl+C
signal.signal(signal.SIGTERM, shutdown_handler)  # Termination signal

try:
    loop.run_until_complete(main())
except KeyboardInterrupt:
    shutdown_handler(None, None)
finally:
    socket.close()
    loop.close()
