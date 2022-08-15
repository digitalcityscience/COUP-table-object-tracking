import asyncio
import socket

from marker import Markers
from camera import poll_frame_data
from tracker import track, track_v2
from time import time_ns

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
        markers_holder.addMarkers(track_v2(frame))
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
        markers_holder.addMarkers(track_v2(frame))
        if (time_ns() - last_sent > 200_000_000):
            markers_json = markers_holder.toJSON()
            print("Sending to unity:", markers_json)
            last_sent = time_ns()
            markers_holder.clear()
            markers_json.encode("utf-8")


loop.run_until_complete(main())
