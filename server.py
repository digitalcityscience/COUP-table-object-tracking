import asyncio
import json
import socket
from typing import Dict

from building import Building, Buildings, printJSON
from camera import poll_frame_data
from tracker import track, track_v2
from time import time_ns

from building import map_detected_buildings
from camera import poll_frame_data
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
    buildings_holder = Buildings()
    last_sent = time_ns()
    for frame in poll_frame_data():
        camera_id, image = frame
        ir_image = sharpen_and_rotate_image(buffer_to_array(image))
        corners, ids, rejectedImgPoints = detect_markers(ir_image)
        buildingDict = map_detected_buildings(camera_id, ids, corners)
        draw_monitor_window(ir_image, corners, rejectedImgPoints, camera_id)
        draw_status_window(buildingDict, camera_id)

        buildings_holder.addBuildings(track_v2(frame))
        if (time_ns() - last_sent > 200_000_000):
            buildings_json = buildings_holder.toJSON()
            print("Sending to unity:", buildings_json)
            last_sent = time_ns()
            buildings_holder.clear()
            await loop.sock_sendall(connection, buildings_json.encode("utf-8"))
            

            
async def test():
    buildings_holder = Buildings()
    last_sent = time_ns()
    for frame in poll_frame_data():
        buildings_holder.addBuildings(track_v2(frame))
        if (time_ns() - last_sent > 200_000_000):
            buildings_json = buildings_holder.toJSON()
            print("Sending to unity:", buildings_json)
            last_sent = time_ns()
            buildings_holder.clear()
            buildings_json.encode("utf-8")


loop.run_until_complete(main())
