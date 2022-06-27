import pyrealsense2 as rs
import numpy as np
import cv2
import cv2.aruco as aruco
import time
import math
import json


import socket
import random


font = cv2.FONT_HERSHEY_SIMPLEX
print("got font")
kernel = np.array([[-1,-1,-1],
                    [-1, 9,-1],
                    [-1,-1,-1]])
buildingDict = {}

gain = 16 #ist nur ein Startwert
exposure = 4000
selectedPoint = 0


pts_src = np.array([[0, 0], [0, 1080], [1080, 0],[1080, 1080]])
pts_dst = np.array([[50, 50], [50, 700], [650, 100],[800, 800]])

class building:
    def __init__(self, id, pos, lastSeen):
        self.id = id
        self.pos = pos
        self.confidence = 0
        self.lastSeen = lastSeen

    def updateConfidence(self, currentLoop):
        self.confidence = currentLoop -self.lastSeen

    def updatePosition(self,pos):
        self.pos = pos
        self.lastSeen = loopcount

    def getConfidence(self):
        return self.confidence

    def getPos(self):
        return self.pos

    def getID(self):
        return self.id

def hsv_to_rgb(h, s, v):  #utility function to convert colorspaces
    if s == 0.0: v*=255; return (v, v, v)
    i = int(h*6.) # XXX assume int() truncates!
    f = (h*6.)-i; p,q,t = int(255*(v*(1.-s))), int(255*(v*(1.-s*f))), int(255*(v*(1.-s*(1.-f)))); v*=255; i%=6
    if i == 0: return (v, t, p)
    if i == 1: return (q, v, p)
    if i == 2: return (p, v, t)
    if i == 3: return (p, q, v)
    if i == 4: return (t, p, v)
    if i == 5: return (v, p, q)

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

    # post_address = 'https://nc.hcu-hamburg.de/cityPyo/updateTableData'
    # r = requests.post(post_address, auth=HTTPBasicAuth('grasbrooker', '%3gZabBC4g3Eu'), json=parentDict, headers={'Content-Type': 'application/json'})


    # if not r.status_code == 200:
    #     print("could not post result to cityPYO", post_address)
    #     print("Error code", r.status_code)
    
    #     #print("Successfully posted to cityPYO", post_address, r.status_code)




    #with open("table_output.json", "w") as write_file:
        #json.dump(parentDict, write_file, indent=4)

def normalizeCorners(corner):
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

    ctrX = np.interp(ctrX,[0,r[2]],[0,1000])
    ctrY = np.interp(ctrY,[0,r[3]],[0,1000])


    returnData = [(int(ctrX),int(ctrY)), angle]
    return returnData

def handleKeypress(key):
    if key == 2424832:
        print("left")
        pts_dst[selectedPoint, 0] += 5
    if key == 2490368:
        print("up")
        pts_dst[selectedPoint, 1] += 5
    if key == 2555904:
        print("right")
        pts_dst[selectedPoint, 0] -= 5
    if key == 2621440:
        print("down")
        pts_dst[selectedPoint, 1] -= 5



#Realsense Config
#--------------------------------------------
pipeline = rs.pipeline()
config = rs.config()
#config.enable_device('001622070380')
config.enable_stream(rs.stream.infrared, 1, 1280 , 800, rs.format.y8,30)
#config.enable_stream(rs.stream.infrared, 2, 1280 , 720, rs.format.y8, 30)
profile = pipeline.start(config)

ir_sensor = profile.get_device().first_depth_sensor()
ir_sensor.set_option(rs.option.emitter_enabled, 0)
ir_sensor.set_option(rs.option.exposure, 3000)
ir_sensor.set_option(rs.option.gain, 16)
IR1_stream = profile.get_stream(rs.stream.infrared, 1) # Fetch stream profile for depth stream
intr = IR1_stream.as_video_stream_profile().get_intrinsics() # Downcast to video_stream_profile and fetch intrinsics
frames = pipeline.wait_for_frames()

# Grab first frame for ROI
#--------------------------------------------
lastFrame = frames.get_infrared_frame()
last_image = np.asanyarray(lastFrame.get_data())
lastFrame = np.zeros((1280,800), np.uint8)

r = cv2.selectROI(last_image)
dx = r[2] - r[0]
dy = r[3] - r[1]
last_image = last_image[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
print(r)
cv2.destroyWindow('ROI selector')
#--------------------------------------------

loopcount = 0
lastUpdatedTime = time.time()
lastSentTime = time.time()




HOST = '127.0.0.1'  # Standard loopback interface address (localhost)
PORT = 8052        # Port to listen on (non-privileged ports are > 1023)

wheelPosX = 0
wheelPosY = 0

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    conn, addr = s.accept()
    with conn:
        print('Connected by', addr)
        try:
            while True:
                if (time.time() - lastSentTime) > 0.05:
                    lastSentTime = time.time()
                    jsonDict = {"500": [wheelPosX,wheelPosY,0]}
                    jsonString= json.dumps(jsonDict)
                    print(jsonString)

                    b = jsonString.encode('utf-8')

                    print("Sent data!")  
                    lastSentTime = time.time()
                    conn.sendall(b)


                if gain > 32: #neuer Loop
                    gain = 16
                    loopcount += 1
                else:
                    try:
                        #ir_sensor.set_option(rs.option.gain, gain)
                        gain += 4
                        #print("TTS: ",(time.time() - start_time) * 1000) # F^^
                    except:
                        print("ERROR FROM GAIN HANDLER")
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
                    if buildingDict[x].getConfidence() > 3: #if not found after 2 loops, discard
                        buildingDict.pop(x)


                ir_image = np.asanyarray(ir_data.get_data())
                ir_image = ir_image[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
                ir_image = cv2.filter2D(ir_image, -1, kernel)
                ir_image = cv2.cvtColor(ir_image,cv2.COLOR_GRAY2BGR)


                aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_1000)
                parameters = aruco.DetectorParameters_create()


                parameters.maxMarkerPerimeterRate = 0.15
                parameters.minMarkerPerimeterRate =0.07
                parameters.polygonalApproxAccuracyRate = 0.03

                parameters.minOtsuStdDev = 2.0
                parameters.perspectiveRemovePixelPerCell = 10
                parameters.perspectiveRemoveIgnoredMarginPerCell = 0.13
                parameters.errorCorrectionRate = 0.3

                parameters.adaptiveThreshWinSizeMin = 3
                parameters.adaptiveThreshWinSizeMax = 23
                parameters.adaptiveThreshWinSizeStep = 5
                parameters.adaptiveThreshConstant = 7

                # ir_image = np.hstack((ir_image,ir_image))
                # ir_image = np.vstack((ir_image,ir_image))
                corners, ids, rejectedImgPoints  = aruco.detectMarkers(ir_image, aruco_dict, parameters=parameters)


                if ids is not None:
                    for i in range(0,len(ids)):
                        cKey = int(ids[i])
                        if cKey is not 0:
                            pos = normalizeCorners(corners[i])
                            if cKey not in buildingDict:
                                buildingDict[cKey] = building(int(ids[i]), pos, loopcount)
                            else:
                                lastPos = buildingDict[cKey].getPos()
                                diff = np.subtract(pos[0], lastPos[0])
                                absDiff = abs(np.sum(diff))
                                if absDiff > 2:
                                    buildingDict[cKey].updatePosition(pos)
                                else:
                                    buildingDict[cKey].updatePosition(lastPos)

                status = np.zeros((800,320,3), np.uint8)
                gui = np.zeros((1080,1080,3), np.uint8)


                for i in range(0,gui.shape[0], 10):
                    gui = cv2.line(gui,(0,i), (gui.shape[1], i), (64,64,64), 2)
                    i += 10
                for j in range(0,gui.shape[1],10):
                    gui = cv2.line(gui,(j,0), (j,gui.shape[1]), (64,64,64), 2)
                    j += 10



                    for point in pts_src:
                        gui = cv2.circle(gui,tuple(point) , 10, (255,255,255), thickness=-10, lineType=8, shift=0)
                    gui = cv2.circle(gui,tuple(pts_src[selectedPoint]) , 50, (0,0,255), thickness=-10, lineType=8, shift=0)



                ir_image = aruco.drawDetectedMarkers(ir_image, corners, borderColor = (0,255,0))
                ir_image = aruco.drawDetectedMarkers(ir_image, rejectedImgPoints, borderColor = (0,0,255))

                for i in buildingDict:
                    id = buildingDict[i].getID()
                    pos =  buildingDict[i].getPos()
                    angle = pos[1]
                    ctr = pos[0]

                    color = np.array(hsv_to_rgb(((angle * 57) % 360)/360,1,1))
                    c1 = tuple([int(x) for x in color])

                    rw = 30
                    rectPoints = [[-rw, -rw], [rw,-rw],[rw,rw], [-rw ,+rw]]

                    for i in range(0,4):
                        rectPoints[i] = rotate(rectPoints[i], angle + math.pi / 4)
                        mappedX = int(np.interp(ctr[0],[0,1000],[0,gui.shape[1]]))
                        mappedY = int(np.interp(ctr[1],[0,1000],[0,gui.shape[0]]))
                        rectPoints[i] = translate(rectPoints[i],(mappedX,mappedY))

                    rectPoints = np.asanyarray(rectPoints)

                    nmbr = np.zeros((40,100,3), np.uint8)
                    title = np.zeros((40,100,3), np.uint8)
                    nmbr = cv2.putText(nmbr, str(id), (0,20), font,0.8,(255,255,255),2)
                    nmbr = cv2.flip(nmbr,0)
                    title = gui[mappedY:mappedY+title.shape[0], mappedX:mappedX+title.shape[1]]
                    try:
                        title = np.maximum(title,nmbr)
                        mappedY += 50
                        gui[mappedY:mappedY+title.shape[0], mappedX:mappedX+title.shape[1]] = title
                    except:
                        continue

                    #gui = cv2.fillPoly(gui,np.int32([rectPoints]),c1)
                    gui = cv2.polylines(gui,np.int32([rectPoints]),True,(255,255,255),2)
                    #gui = cv2.circle(gui,ctr , 2, (255,255,255), thickness=-2, lineType=8, shift=0)


                gui = cv2.flip(gui, 0)
                gui = cv2.rotate(gui,cv2.ROTATE_90_COUNTERCLOCKWISE)
                h, state = cv2.findHomography(pts_src, pts_dst)
                gui = cv2.warpPerspective(gui, h, (gui.shape[1],gui.shape[0]))

                statusX = 50
                #fps = 1.0 / (time.time() - start_time)
                #cv2.putText(status, str(int(fps)), (10, 750), font,0.8,(255,255,255),1)

                for x in buildingDict:
                    ctr = buildingDict[x].getPos()[0]
                    id = buildingDict[x].getID()
                    conf = buildingDict[x].getConfidence()

                    cv2.putText(status, str(id), (30, statusX), font,0.8,(255,255,255),1)
                    cv2.putText(status, str(ctr),(100,statusX),font,0.6,(255,255,255),1)
                    cv2.putText(status, str(conf),(270,statusX),font,0.6,(255,255,255),1)
                    statusX += 35

                cv2.namedWindow('GUI', cv2.WINDOW_AUTOSIZE)
                cv2.namedWindow('IR', cv2.WINDOW_AUTOSIZE)
                cv2.namedWindow('Status', cv2.WINDOW_AUTOSIZE)

                # cv2.namedWindow('GUI',cv2.WND_PROP_FULLSCREEN)
                # cv2.setWindowProperty('GUI', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                #cv2.namedWindow('IR', cv2.WINDOW_NORMAL)
                #cv2.imshow('RealSense', gui)
                #ir_image = cv2.resize(ir_image, (1280, 800))
                cv2.imshow('GUI', gui)
                cv2.imshow('IR', ir_image)
                #cv2.imshow('IR', ir_image)
                cv2.imshow('Status', status)
                #cv2.resizeWindow('IR', 400,400)
                #


                if 6 in buildingDict.keys():
                    wheelPosX = buildingDict[6].getPos()[0][0]
                    wheelPosY = buildingDict[6].getPos()[0][1]


                key = cv2.waitKeyEx(1)

                if key == 32:
                        # printJSON(buildingDict)
                        # print("JSON dumped")

                        print("Homography dumped") 
                        np.savetxt("homography.txt", pts_dst, fmt="%s")

                if key == ord('l'):
                    pts_dst = np.loadtxt(open("homography.txt"))
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


        finally:
            # Stop streaming
            pipeline.stop()
