# CityScope object tracking

## How to run
- Install requirements
- Run server.py 
- Connect to the opened websocket to read results. (By running the COUP-TangibleTable unity project)
- A visual debug monitor will show you what the camera sees and which markers are recognized.

Depending on your light conditions you might need to choose different settings for exposure and gain in the realtsense/realsense_device_manager.py 


![image](https://user-images.githubusercontent.com/4631906/219411610-2bd0850a-6c8e-4b90-a0ca-70bdeb26b108.png)

### Sources
 - https://github.com/IntelRealSense/librealsense/tree/master/wrappers/python
 - [realsense API](https://intelrealsense.github.io/librealsense/python_docs/_generated/pyrealsense2.html#module-pyrealsense2)
 - [aruco opencv](https://docs.opencv.org/4.x/d9/d6a/group__aruco.html#gab9159aa69250d8d3642593e508cb6baa)
