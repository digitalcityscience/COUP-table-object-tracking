import asyncio
import json

import websockets

WS_SERVER_SETTINGS = ("localhost", 8053)

# Sample calibration points to exercise the map_calibration message; adjust as needed.
# Corners span roughly 1000m (east-west) by 500m (north-south) around Hamburg, Germany.
SAMPLE_CALIBRATION_POINTS = [
    {"pixel_position": [0, 0], "lat_lon_position": [53.5511, 9.9937]},
    {"pixel_position": [1000, 0], "lat_lon_position": [53.5511, 10.0088]},
    {"pixel_position": [1000, 800], "lat_lon_position": [53.5466, 10.0088]},
    {"pixel_position": [0, 800], "lat_lon_position": [53.5466, 9.9937]},
]


async def receive_messages(websocket):
    async for message in websocket:
        print(f"Received: {message}")


async def send_calibration(websocket):
    payload = {"type": "map_calibration", "points": SAMPLE_CALIBRATION_POINTS}
    print(f"Sending map_calibration: {payload}")
    await websocket.send(json.dumps(payload))


async def main():
    uri = f"ws://{WS_SERVER_SETTINGS[0]}:{WS_SERVER_SETTINGS[1]}"
    print(f"Connecting to websocket on: {uri}")

    async with websockets.connect(uri) as websocket:
        print("Connected to server!")
        await send_calibration(websocket)
        try:
            await receive_messages(websocket)
        finally:
            print("The server closed the connection")


asyncio.run(main())
