from typing import Dict, List
from building import Building, add_detected_buildings_to_dict, printJSON
from detection import detect_markers, normalizeCorners
from hud import draw_monitor_window, draw_status_window, handle_key_presses
from image import buffer_to_array, sharpen_and_rotate_image
import pyrealsense2 as rs
import numpy as np
import time
import math
import json
import socket

buildingDict:Dict[int,Building] = {}

loop = 16
exposure = 8000

#Realsense Config
#--------------------------------------------
pipeline = rs.pipeline()
config = rs.config()
#config.enable_device('00162207038+0')
config.enable_stream(rs.stream.infrared, 1, 1280 , 800, rs.format.y8,30)
#config.enable_stream(rs.stream.infrared, 2, 1280 , 720, rs.format.y8, 30)
profile = pipeline.start(config)

ir_sensor = profile.get_device().first_depth_sensor()
ir_sensor.set_option(rs.option.emitter_enabled, 0)
ir_sensor.set_option(rs.option.exposure, exposure)
ir_sensor.set_option(rs.option.gain, 16)
IR1_stream = profile.get_stream(rs.stream.infrared, 1) # Fetch stream profile for depth stream
intr = IR1_stream.as_video_stream_profile().get_intrinsics() # Downcast to video_stream_profile and fetch intrinsics
frames = pipeline.wait_for_frames()

loopcount = 0
lastUpdatedTime = time.time()
lastSentTime = time.time()


HOST = '127.0.0.1'  # Standard loopback interface address (localhost)
PORT = 8052        # Port to listen on (non-privileged ports are > 1023)

wheelPosX = 0
wheelPosY = 0


def send_detected_buildings(socket:socket.socket, buildingDict:Dict[int,Building], last_send_time:float) -> None:
    if (time.time() - last_send_time) > 0.05:
        jsonString= json.dumps(printJSON(buildingDict))
        socket.sendall(jsonString.encode('utf-8'))

while True:
    print("starting socket")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()

        conn, addr = s.accept()
        with conn:
            print('Connected by', addr)

            while conn:
                    if loop > 32: #neuer Loop
                        loop = 16
                        loopcount += 1
                    else:
                        try:
                            #ir_sensor.set_option(rs.option.gain, gain)
                            loop += 4
                            #print("TTS: ",(time.time() - start_time) * 1000) # F^^
                        except:
                            loopcount -= 1

                    try:
                        frames = pipeline.wait_for_frames()
                        ir_data = frames.get_infrared_frame()
                    except:
                        "Frame Aquisition Error"
                        continue

                    if not ir_data:
                        continue

                    for x in list(buildingDict):
                        buildingDict[x].updateConfidence(loopcount)
                        if buildingDict[x].getConfidence() > 5: #if not found after 5 loops, discard
                            buildingDict.pop(x)
                            
                    ir_image = sharpen_and_rotate_image(buffer_to_array(ir_data.get_data()))
                    corners, ids, rejectedImgPoints  = detect_markers(ir_image)

                    add_detected_buildings_to_dict(ids, corners, loopcount, buildingDict)

                    try:
                        send_detected_buildings(conn, buildingDict, lastSentTime)
                        lastSentTime = time.time()
                    except:
                        break

                    draw_monitor_window(ir_image,corners, rejectedImgPoints)
                    draw_status_window(buildingDict)
                    handle_key_presses()
            s.close()
            print("conn was aparrently closed!")
                    




