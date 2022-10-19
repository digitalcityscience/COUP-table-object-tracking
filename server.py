import asyncio
import socket

from marker import Markers, map_detected_markers
from camera import poll_frame_data
from tracker import track, track_v2
from time import time_ns
from detection import detect_markers
from hud import draw_monitor_window, draw_status_window
from image import buffer_to_array, sharpen_and_rotate_image

socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
SERVER_SETTINGS = ("localhost", 8052)
print(f"Listening to socket connections on: {SERVER_SETTINGS}")
socket.bind(SERVER_SETTINGS)
socket.listen(1)
socket.setblocking(False)
loop = asyncio.new_event_loop()


async def main():
    while True:
        connection, client_address = await loop.sock_accept(socket)
        print(f"Connection from: {client_address}")
        loop.create_task(send_tracking_matches(connection))


async def send_tracking_matches(connection):
    markers_holder = Markers()
    last_sent = time_ns()
    for frame in poll_frame_data():
        camera_id, image = frame
        ir_image = sharpen_and_rotate_image(buffer_to_array(image))
        corners, ids, rejectedImgPoints = detect_markers(ir_image)
        buildingDict = map_detected_markers(camera_id, ids, corners)
        draw_monitor_window(ir_image, corners, rejectedImgPoints, camera_id)
        draw_status_window(buildingDict, camera_id)

        markers_holder.updateMarkers(track_v2(frame))
        if (time_ns() - last_sent > 200_000_000):
            markers_json = markers_holder.toJSON()
            print("Sending to unity:", markers_json)
            last_sent = time_ns()
            markers_holder.clear()
            await loop.sock_sendall(connection, markers_json.encode("utf-8"))
            

            
async def test():
    markers_holder = Markers()
    last_sent = time_ns()
    for frame in poll_frame_data():
        markers_holder.updateMarkers(track_v2(frame))
        if (time_ns() - last_sent > 200_000_000):
            markers_json = markers_holder.toJSON()
            print("Sending to unity:", markers_json)
            last_sent = time_ns()
            markers_holder.clear()
            markers_json.encode("utf-8")


loop.run_until_complete(main())
