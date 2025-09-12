import asyncio
import json
import socket
import signal
import sys
import time

from marker import Markers, map_detected_markers
from tracker import track_v2
from time import time_ns
from detection import detect_markers
from hud import draw_monitor_window, draw_status_window
from find_calibration_markers import save_calibration_markers, check_calibration_exists
from camera_stitching import setup_camera_transforms, process_and_join_streams
import cv2

from collections import defaultdict
from typing import Dict, List
import os
from datetime import datetime


def load_calibration_markers(file_path: str) -> Dict:
    """Load calibration markers from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

# Global variable for stitching setup
stitching_setup = None

socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
SERVER_SETTINGS = ("localhost", 8052)
print(f"Listening to socket connections on: {SERVER_SETTINGS}")
socket.bind(SERVER_SETTINGS)
socket.listen(1)
socket.setblocking(False)
loop = asyncio.new_event_loop()

def prompt_recalibration(timeout=10):
    print("Existing calibration found.")
    print("Press 1 to recalibrate within {} seconds...".format(timeout))
    
    start_time = time.time()
    while True:
        if time.time() - start_time > timeout:
            print("No input received. Continuing with existing calibration.")
            return False
        user_input = input()
        if user_input == '1':
            print("Recalibrating...")
            return True


def initial_calibration():

    # Step 1: Run calibration setup
    print("Step 1: Running calibration setup...")
    if check_calibration_exists():
        print("Existing calibration found.")
        if prompt_recalibration(timeout=5):
            print("OVERRIDING EXISITING CALIBRATION")
            save_calibration_markers()

    save_calibration_markers()
    
        

def initialize_camera_stitching():
    """Initialize camera stitching setup"""
    global stitching_setup
    
    print("=== Initializing Camera Stitching System ===")
    
       # Step 1: Setup camera transforms
    print("Step 2: Setting up camera transforms...")
    stitching_setup = setup_camera_transforms(load_calibration_markers("calibration_markers.json"))
    
    # Step 2: try if the setup works
    try:
        # Get one stitched frame as a sample
        for _stitched_image in process_and_join_streams(stitching_setup):
            print("✓ Abble to process and stitch camera streams")
            break  # Only export one sample, then continue
    except Exception as e:
        print(f"Warning: Could process and stitch camera streams: {e}")
    
    return stitching_setup
        


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
