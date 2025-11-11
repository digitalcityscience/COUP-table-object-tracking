import asyncio
import json
import socket
import signal
import sys
import time

from marker import Markers, map_detected_markers
from time import time_ns
from detection import detect_markers
from hud import draw_monitor_window, draw_status_window
from calibration_handler  import load_calibration_markers, run_initial_calibration_if_needed
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
              


async def main():
    # Runs the initial table calibration setup if no calibration file is found
    run_initial_calibration_if_needed()
    # Initialize camera stitching system at startup
    global stitching_setup
    stitching_setup = setup_camera_transforms(load_calibration_markers("calibration_markers.json"))
    print("waiting for client to connect")
    
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
        buildingDict = map_detected_markers("000", ids, corners)
        
        # Show stitched result with markers
        draw_monitor_window(stitched_image, corners, rejectedImgPoints, "000")
        draw_status_window(buildingDict, "000")
        
        markers_holder.addMarkers(list(buildingDict.values())) 
        
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
