import argparse
import asyncio
import json
import logging
import queue as queue_module
import socket as socket_module
import signal
import sys
import threading

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

# Basemap homography computed from the web client's map_calibration message.
# None until the web client sends calibration points, in which case tracking
# matches are sent as GeoJSON (see handle_web_client); otherwise a plain
# marker dict is sent.
basemap_homography: BasemapHomography | None = None

# Raw TCP socket used only for the "unity" client. Created in setup_pixel_socket().
pixel_socket = None
PIXEL_SOCKET_SETTINGS = ("localhost", 8052)

loop = asyncio.new_event_loop()

WEB_WS_HOST = "0.0.0.0"
WEB_WS_PORT = 8053

# Marker snapshots (dicts) produced by the detection thread, consumed by
# whichever client (unity/web) is sending tracking updates. Bounded to 1 so
# stale snapshots get replaced instead of piling up if a consumer falls behind.
tracking_queue: "queue_module.Queue[dict]" = queue_module.Queue(maxsize=1)


def _parse_calibration_points(raw_points: list) -> list[BasemapCalibrationPoint]:
    return [
        BasemapCalibrationPoint(
            pixel_position=tuple(point["pixel_position"]),
            lat_lon_position=tuple(point["lat_lon_position"]),
        )
        for point in raw_points
    ]


def _detection_worker(interval_ns: int = 200_000_000):
    """Runs on a dedicated background thread (NOT the asyncio event loop).

    Continuously captures/stitches/detects camera frames (all blocking,
    CPU/hardware-bound work) and pushes an accumulated marker snapshot onto
    `tracking_queue` every `interval_ns`. Keeping this off the event loop
    thread means network I/O (receiving/sending on sockets/websockets) is
    never blocked by camera polling or cv2 rendering.
    """
    markers_holder = Markers()
    last_sent = time_ns()

    for stitched_image in process_and_join_streams(stitching_setup):
        corners, ids, rejectedImgPoints = detect_markers(stitched_image)  # runs detection.
        buildingDict = map_detected_markers("000", ids, corners)

        draw_monitor_window(stitched_image, corners, rejectedImgPoints, "000")
        draw_status_window(buildingDict, "000")

        markers_holder.addMarkers(list(buildingDict.values()))

        if (time_ns() - last_sent > interval_ns):
            snapshot = markers_holder.toDict()

            # Keep only the freshest snapshot if the consumer is falling behind.
            if tracking_queue.full():
                try:
                    tracking_queue.get_nowait()
                except queue_module.Empty:
                    pass
            tracking_queue.put_nowait(snapshot)

            last_sent = time_ns()
            markers_holder.clear()


def start_detection_thread():
    threading.Thread(target=_detection_worker, name="detection-worker", daemon=True).start()


async def stream_tracking_updates(send):
    """Consume marker snapshots from `tracking_queue` and forward each via `send`.

    `send` is an async callable taking a marker snapshot dict; it should
    format and transmit it to the client.
    """
    while True:
        snapshot = await asyncio.to_thread(tracking_queue.get)
        await send(snapshot)


async def handle_web_client(websocket):
    """Handle a web client websocket connection: receive calibration updates
    and stream tracking results back.

    Expected incoming message schema:
        {"type": "map_calibration", "points": [{"pixel_position": [x, y], "utm_position": [x, y]}, ...]}
        {"type": "building_calibration", ...}  # not implemented yet

    Outgoing tracking messages are sent as GeoJSON once a basemap homography
    is available (i.e. after a map_calibration message has been received),
    otherwise as a plain dict of markers.
    """
    peer = websocket.remote_address
    print(f"Web client connected: {peer}")

    async def receive_messages():
        global basemap_homography
        async for message in websocket:
            try:
                payload = json.loads(message)
            except json.JSONDecodeError:
                print(f"Could not parse message from web client: {message!r}")
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
                print(f"Unknown message type from web client: {message_type!r}")

    async def send_tracking_updates():
        async def send(snapshot: dict):
            global basemap_homography

            # Once the web client has sent map calibration points, convert
            # marker positions to GeoJSON; until then, send a plain dict
            # of markers.
            if basemap_homography is not None:
                markers_json = json.dumps(markers_json_to_geojson(snapshot, basemap_homography))
            else:
                markers_json = json.dumps(snapshot)

            print("Sending to web client:", markers_json)
            await websocket.send(markers_json)

        await stream_tracking_updates(send)  # 200ms interval between updates

    try:
        await asyncio.gather(receive_messages(), send_tracking_updates())
    except websockets.exceptions.ConnectionClosed as exc:
        print(f"Web client connection closed: {exc}")
    except Exception:
        print(f"Web client handler crashed for {peer}:")
        import traceback
        traceback.print_exc()
    finally:
        print(f"Web client disconnected: {peer}")


async def run_web_server():
    async with websockets.serve(handle_web_client, WEB_WS_HOST, WEB_WS_PORT):
        print(f"Listening for web client connections on {WEB_WS_HOST}:{WEB_WS_PORT}")
        await asyncio.Future()  # run forever


def setup_pixel_socket():
    global pixel_socket
    pixel_socket = socket_module.socket(socket_module.AF_INET, socket_module.SOCK_STREAM)
    print(f"Listening to socket connections on: {PIXEL_SOCKET_SETTINGS}")
    pixel_socket.bind(PIXEL_SOCKET_SETTINGS)
    pixel_socket.listen(1)
    pixel_socket.setblocking(False)


async def run_pixel_socket_server():
    while True:
        connection, client_address = await loop.sock_accept(pixel_socket)
        print(f"Unity client connected from: {client_address}")
        loop.create_task(send_tracking_matches_unity(connection))


async def send_tracking_matches_unity(connection):
    """Stream marker JSON to a Unity client.

    Always sends the raw marker JSON string, regardless of whether a basemap
    homography has been configured; the Unity client does not send anything
    back to the server.
    """
    async def send(snapshot: dict):
        markers_json = json.dumps(snapshot)
        print("Sending to Unity client:", markers_json)
        await loop.sock_sendall(connection, markers_json.encode("utf-8"))

    await stream_tracking_updates(send)


async def main(client: str):
    # Runs the initial table calibration setup if no calibration file is found
    run_initial_calibration_if_needed()
    # Initialize camera stitching system at startup
    global stitching_setup
    stitching_setup = setup_camera_transforms(load_calibration_markers("calibration_markers.json"))

    # Detection/stitching runs on its own thread so it never blocks the
    # asyncio event loop (which handles client I/O).
    start_detection_thread()

    print(f"waiting for {client} client to connect")

    if client == "unity":
        setup_pixel_socket()
        await run_pixel_socket_server()
    else:
        await run_web_server()


def shutdown_handler(sig, frame):
    """Handle graceful shutdown"""
    print("\nShutting down server...")
    if pixel_socket is not None:
        print("Closing socket...")
        pixel_socket.close()
    print("Stopping event loop...")
    loop.stop()
    sys.exit(0)


def parse_args():
    parser = argparse.ArgumentParser(description="COUP table object tracking server")
    parser.add_argument(
        "--client",
        choices=["unity", "web"],
        default="web",
        required=False,
        help="Which client to serve: 'unity' (raw TCP socket, marker JSON) or 'web' (websocket, geojson/dict)",
    )
    return parser.parse_args()


args = parse_args()

logging.basicConfig(level=logging.INFO)

# Register signal handlers
signal.signal(signal.SIGINT, shutdown_handler)  # Ctrl+C
signal.signal(signal.SIGTERM, shutdown_handler)  # Termination signal

try:
    loop.run_until_complete(main(args.client))
except KeyboardInterrupt:
    shutdown_handler(None, None)
finally:
    if pixel_socket is not None:
        pixel_socket.close()
    loop.close()
