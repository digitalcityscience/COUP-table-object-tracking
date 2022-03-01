from random import randrange
from time import sleep
import pyrealsense2 as rs
from realsense.realsense_device_manager import DeviceManager
import cv2
import numpy

def write_to_file(buffer, file_name:str = "output.png"):
  cv2.imwrite(f'./output/{file_name}', numpy.asanyarray(buffer))


if __name__ == "__main__":
    device_manager = None
    try:
      config = rs.config()
      config.enable_stream(rs.stream.infrared, 1, 1280, 800, rs.format.y8, 30)
      device_manager = DeviceManager(rs.context(), config)
      device_manager.enable_all_devices()
      frames = []
      for k in range(5):
          frames = device_manager.poll_frames()
          print(frames)
          frame_id = randrange(1000)
          for camera_id in frames:
            frame = frames[camera_id]
            frame_value = list(frame.values())[0]
            frame_data = frame_value.get_data()
            print( frame_data )
            write_to_file(frame_data, f'output_{frame_id}_cid_{camera_id[0][-3:]}.png')
          print("Done")
          sleep(10)
    finally:
      if (device_manager):
        device_manager.disable_streams()