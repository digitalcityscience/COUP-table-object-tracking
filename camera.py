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
      for k in range(15):
          frames = device_manager.poll_frames()
      print(frames)
      a = frames['001622070721', 'D400']
      b = list(a.values())[0]
      c = b.get_data()
      print( c )
      write_to_file(c)
      print("Done")
    finally:
      if (device_manager):
        device_manager.disable_streams()