import asyncio
import json
import socket
import signal
import sys

from marker import Markers, map_detected_markers
from time import time_ns
from detection import detect_markers
from hud import draw_monitor_window, draw_status_window
from calibration_handler  import load_calibration_markers, run_initial_calibration_if_needed
from camera_stitching import setup_camera_transforms, process_and_join_streams
from pixel_to_utm import BasemapCalibrationPoint, BasemapHomography, create_basemap_homography
from table_to_geojson import markers_json_to_geojson
import websockets


# Global variable for stitching setup
stitching_setup = None

# Basemap homography computed from the frontend's map_calibration message.
# None until the frontend sends calibration points, in which case tracking
# matches are sent as raw pixel-space marker JSON (see send_tracking_matches).
basemap_homography: BasemapHomography | None = None

socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
SERVER_SETTINGS = ("localhost", 8052)
print(f"Listening to socket connections on: {SERVER_SETTINGS}")
socket.bind(SERVER_SETTINGS)
socket.listen(1)
socket.setblocking(False)
loop = asyncio.new_event_loop()  

CALIBRATION_WS_HOST = "0.0.0.0"
CALIBRATION_WS_PORT = 8053


def _parse_calibration_points(raw_points: list) -> list[BasemapCalibrationPoint]:
    return [
        BasemapCalibrationPoint(
            pixel_position=tuple(point["pixel_position"]),
            utm_position=tuple(point["utm_position"]),
        )
        for point in raw_points
    ]


async def handle_calibration_client(websocket):
    """Handle a frontend websocket connection sending calibration updates.

    Expected message schema:
        {"type": "map_calibration", "points": [{"pixel_position": [x, y], "utm_position": [x, y]}, ...]}
        {"type": "building_calibration", ...}  # not implemented yet
    """
    global basemap_homography
    peer = websocket.remote_address
    print(f"Frontend connected on calibration websocket: {peer}")

    try:
        async for message in websocket:
            try:
                payload = json.loads(message)
            except json.JSONDecodeError:
                print(f"Could not parse calibration message: {message!r}")
                continue

            message_type = payload.get("type")

            if message_type == "map_calibration":
                try:
                    calibration_points = _parse_calibration_points(payload["points"])
                    basemap_homography = create_basemap_homography(calibration_points)
                    print(f"Updated basemap homography from {len(calibration_points)} calibration points")
                except (KeyError, ValueError) as exc:
                    print(f"Invalid map_calibration payload: {exc}")

            elif message_type == "building_calibration":
                print("building_calibration not implemented yet")

            else:
                print(f"Unknown calibration message type: {message_type!r}")

    except websockets.exceptions.ConnectionClosed as exc:
        print(f"Calibration websocket connection closed: {exc}")
    finally:
        print(f"Frontend disconnected from calibration websocket: {peer}")


async def run_calibration_websocket_server():
    async with websockets.serve(handle_calibration_client, CALIBRATION_WS_HOST, CALIBRATION_WS_PORT):
        print(f"Listening for calibration websocket connections on {CALIBRATION_WS_HOST}:{CALIBRATION_WS_PORT}")
        await asyncio.Future()  # run forever


async def run_pixel_socket_server():
    while True:
        connection, client_address = await loop.sock_accept(socket)
        print(f"Connection from: {client_address}")
        loop.create_task(send_tracking_matches(connection))


async def main():
    # Runs the initial table calibration setup if no calibration file is found
    run_initial_calibration_if_needed()
    # Initialize camera stitching system at startup
    global stitching_setup
    stitching_setup = setup_camera_transforms(load_calibration_markers("calibration_markers.json"))
    print("waiting for client to connect")

    await asyncio.gather(
        run_pixel_socket_server(),
        run_calibration_websocket_server(),
    )


async def send_tracking_matches(connection):
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
            # Once the frontend has sent map calibration points, convert marker
            # positions to GeoJSON; until then, fall back to the original raw
            # marker JSON string for backwards compatibility [Unity]
            if basemap_homography is not None:
                markers_json = json.dumps(markers_json_to_geojson(markers_holder.toDict(), basemap_homography))
            else:
                markers_json = markers_holder.toJSON()

            print("Sending to websocket:", markers_json)
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
