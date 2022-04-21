import asyncio
import json
import socket
from typing import Dict

from building import Building, printJSON
from camera import poll_frame_data
from tracker import track

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
    for frame in poll_frame_data():
        buildingDict: Dict[int, Building] = {}
        track(frame, buildingDict)
        jsonString = json.dumps(printJSON(buildingDict))
        print("Sending to unity:", jsonString)
        await loop.sock_sendall(connection, jsonString.encode("utf-8"))


loop.run_until_complete(main())
