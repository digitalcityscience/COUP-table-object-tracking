from typing import Dict, List
from building import Building
from detection import detect_markers
from hud import draw_monitor_window, draw_status_window
from image import buffer_to_array, sharpen_and_rotate_image
import pyrealsense2 as rs
import numpy as np
import cv2
import cv2.aruco as aruco
import time
import math
import json
import socket

kernel = np.array([[-1,-1,-1],
                    [-1, 9,-1],
                    [-1,-1,-1]])
buildingDict:Dict[int,Building] = {}

loop = 16
exposure = 8000
selectedPoint = 0

#[width, height]
pts_src = np.array([[0, 0], [0, 800], [1280, 0],[1280, 800]])
pts_dst = np.array([[0, 0], [0, 1000], [1000, 0],[1000, 1000]])


def rotate(xy, theta):
    # https://en.wikipedia.org/wiki/Rotation_matrix#In_two_dimensions
    cos_theta, sin_theta = math.cos(theta), math.sin(theta)

    return (
        xy[0] * cos_theta - xy[1] * sin_theta,
        xy[0] * sin_theta + xy[1] * cos_theta
    )

def translate(xy, offset):
    return xy[0] + offset[0], xy[1] + offset[1]

def printJSON(data):
    jsonDict = {}
    parentDict ={}

    for i in data:
        jsonDict[i] = data[i].getPos()

    parentDict["table_state"] = jsonDict
    return jsonDict

def normalizeCorners(corner) -> List[float]:
    coords = corner
    pts = coords.reshape((-1,1,2))

    p1 = tuple(pts[0][0])
    p4 = tuple(pts[2][0])

    ctrX = (p1[0] + p4[0]) / 2
    ctrY = (p1[1] + p4[1]) / 2

    dx = p1[0] - ctrX
    dy = p1[1] - ctrY

    angle = math.atan2(dy,dx)
    angleDeg = math.degrees(angle)

    ctrX = np.interp(ctrX,[0,10000],[0,10000])
    ctrY = np.interp(ctrY,[0,10000],[0,10000])

    return [int(ctrX), int(ctrY), angleDeg]

def handleKeypress(key):
    if key == 2424832:
        print("left")
        pts_src[selectedPoint, 0] += 1
    if key == 2490368:
        print("up")
        pts_src[selectedPoint, 1] += 1
    if key == 2555904:
        print("right")
        pts_src[selectedPoint, 0] -= 1
    if key == 2621440:
        print("down")
        pts_src[selectedPoint, 1] -= 1



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

pts_src = np.loadtxt(open("homography.txt"))
print("Homography loaded")


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

                    if ids is not None:
                        for i in range(0,len(ids)):
                            markerID = int(ids[i])

                            if markerID is not 500:
                                pos = normalizeCorners(corners[i])

                                if markerID not in buildingDict:
                                    buildingDict[markerID] = Building(int(ids[i]), pos, loopcount)
                                else:
                                    buildingDict[markerID].updatePosition(pos, loopcount)

                    try:
                        send_detected_buildings(conn, buildingDict, lastSentTime)
                        lastSentTime = time.time()
                    except:
                        break

                    draw_monitor_window(ir_image,corners, rejectedImgPoints)
                    draw_status_window(buildingDict)

                    key = cv2.waitKeyEx(1)

                    if key == 32:
                            print("Homography dumped") 
                            np.savetxt("homography.txt", pts_src, fmt="%s")

                    if key == ord('l'):
                        pts_src = np.loadtxt(open("homography.txt"))
                        print("Homography loaded")
                    if key == ord('1'):
                        selectedPoint = 0
                        print("point 1")
                    if key == ord('2'):
                        selectedPoint = 1
                        print("point 2")
                    if key == ord('3'):
                        selectedPoint = 2
                        print("point 3")
                    if key == ord('4'):
                        selectedPoint = 3
                        print("point 4")
                    elif key is not -1:
                            handleKeypress(key)

            s.close()
            print("conn was aparrently closed!")
                    




