import time
from functools import lru_cache
from time import sleep
from typing import Any, Iterable, Tuple

import pyrealsense2 as rs

from image import write_to_file
from realsense.realsense_device_manager import DeviceManager

FRAMES_PER_SECOND = 15

Frame = Iterable[Tuple[int, Any]]


@lru_cache(1)
def get_device_manager() -> DeviceManager:
    print("initializing realsense device manager")
    config = rs.config()
    config.enable_stream(
        rs.stream.infrared, 1, 1280, 800, rs.format.y8, FRAMES_PER_SECOND
    )
    device_manager = DeviceManager(rs.context(), config)
    device_manager.enable_all_devices()
    print(f"Active device serial numbers: {device_manager.get_enabled_devices_ids()}")
    return device_manager


def poll_frame_data() -> Frame:
    device_manager = None
    try:
        device_manager = get_device_manager()
        frames = []
        while True:
            frames = device_manager.poll_frames()
            for camera_id in frames:
                frame = frames[camera_id]
                frame_value = list(frame.values())[0]
                frame_data = frame_value.get_data()
                short_camera_id = camera_id[0][-3:]
                yield short_camera_id, frame_data
    finally:
        if device_manager:
            device_manager.disable_streams()


def get_latest_frame_data() -> Iterable[Tuple[int, Any]]:
    device_manager = None
    try:
        device_manager = get_device_manager()
        frames = []
        frames = device_manager.poll_frames()
        for camera_id in frames:
            frame = frames[camera_id]
            frame_value = list(frame.values())[0]
            frame_data = frame_value.get_data()
            short_camera_id = camera_id[0][-3:]
            yield short_camera_id, frame_data
    finally:
        if device_manager:
            device_manager.disable_streams()


def write_images():
    device_manager = None
    try:
        device_manager = get_device_manager()
        frames = []
        for k in range(5):
            frames = device_manager.poll_frames()
            print(frames)
            for camera_id in frames:
                frame = frames[camera_id]
                frame_value = list(frame.values())[0]
                frame_data = frame_value.get_data()
                print(frame_data)
                write_to_file(
                    frame_data, f"output_{time.time()}_cid_{camera_id[0][-3:]}.png"
                )
            print("Done")
            sleep(10)
    finally:
        if device_manager:
            device_manager.disable_streams()


if __name__ == "__main__":
    write_images()
    exit()
