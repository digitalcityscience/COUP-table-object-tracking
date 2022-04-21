import json
import socket
import time
from typing import Dict

import reactivex as rx

from building import Building, printJSON
from camera import poll_frame_data
from tracker import track

buildingDict: Dict[int, Building] = {}

loop = 16
exposure = 8000


loopcount = 0
lastUpdatedTime = time.time()
lastSentTime = time.time()


HOST = "127.0.0.1"  # Standard loopback interface address (localhost)
PORT = 8052  # Port to listen on (non-privileged ports are > 1023)


# track frames from all cameras
frames = rx.from_iterable(poll_frame_data())


def send_detected_buildings(
    socket: socket.socket, buildingDict: Dict[int, Building], last_send_time: float
) -> None:
    jsonString = json.dumps(printJSON(buildingDict))
    print("Sending to unity:", jsonString)
    socket.sendall(jsonString.encode("utf-8"))


def track_and_send(frame, connection):
    track(frame, buildingDict)
    send_detected_buildings(connection, buildingDict, time.time())


while True:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        print(f"Starting socket on port: {PORT}")
        s.bind((HOST, PORT))
        s.listen()
        conn, addr = s.accept()
        with conn:
            print("Connected by", addr)
            frames.subscribe(lambda frame: track_and_send(frame, conn))
            s.close()
            print("conn was aparrently closed!")
