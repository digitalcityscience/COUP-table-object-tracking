import math
from typing import Dict
import numpy
import cv2
import cv2.aruco as aruco

from building import Building


def draw_status_window(buildingDict: Dict[int, Building], camera_id: int = 0) -> None:
    status = numpy.zeros((800, 335, 3), numpy.uint8)
    statusY = 50
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(status, 'mId', (30, statusY), font,  0.6, (255, 255, 255), 1)
    cv2.putText(status, 'xPos', (100, statusY), font, 0.6, (255, 255, 255), 1)
    cv2.putText(
            status,
            f'yDeg',
            (150, statusY),
            font,
            0.6,
            (255, 255, 255),
            1,
        )
    cv2.putText(status, '?%', (220, statusY), font, 0.6, (255, 255, 255), 1)
    cv2.putText(status, f'cId', (290, statusY), font, 0.6, (255, 255, 255), 1)
    statusY += 35

    for x in buildingDict:
        ctr = buildingDict[x].getPos()[0]
        deg = buildingDict[x].getPos()[1]
        id = buildingDict[x].getID()
        conf = buildingDict[x].getConfidence()

        cv2.putText(status, str(id), (30, statusY), font, 0.8, (255, 255, 255), 1)
        cv2.putText(status, str(ctr), (100, statusY), font, 0.6, (255, 255, 255), 1)
        cv2.putText(
            status,
            str(math.floor(math.radians(deg))),
            (150, statusY),
            font,
            0.6,
            (255, 255, 255),
            1,
        )
        cv2.putText(status, str(conf), (220, statusY), font, 0.6, (255, 255, 255), 1)
        cv2.putText(status, f'{camera_id}', (290, statusY), font, 0.6, (255, 255, 255), 1)
        statusY += 35
    
    window_name = f'Status_{camera_id}'
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(window_name, status)


def draw_monitor_window(ir_image, corners, rejectedImgPoints, window_id: int = 0) -> None:

    ir_image = aruco.drawDetectedMarkers(ir_image, corners, borderColor=(0, 255, 0))
    ir_image = aruco.drawDetectedMarkers(
        ir_image, rejectedImgPoints, borderColor=(0, 0, 255)
    )
    for i in range(0, ir_image.shape[0], 50):
        ir_image = cv2.line(
            ir_image, (0, i), (ir_image.shape[1], i), (255, 255, 255), 1
        )
        i += 10
    for j in range(0, ir_image.shape[1], 50):
        ir_image = cv2.line(
            ir_image, (j, 0), (j, ir_image.shape[1]), (255, 255, 255), 1
        )
        j += 10
    window_name = f'IR_{window_id}'
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(window_name, ir_image)
    handle_key_presses()


#[width, height]
pts_src = numpy.loadtxt(open("homography.txt"))
if(pts_src is None):
  pts_src = numpy.array([[0, 0], [0, 800], [1280, 0],[1280, 800]])
print("Homography loaded")

def handle_key_presses() -> None:
    key = cv2.waitKeyEx(1)
    selectedPoint = 0
    if key == 32:
        print("Homography dumped")
        numpy.savetxt("homography.txt", pts_src, fmt="%s")
    if key == ord("l"):
        pts_src = numpy.loadtxt(open("homography.txt"))
        print("Homography loaded")
    if key == ord("1"):
        selectedPoint = 0
        print("point 1")
    if key == ord("2"):
        selectedPoint = 1
        print("point 2")
    if key == ord("3"):
        selectedPoint = 2
        print("point 3")
    if key == ord("4"):
        selectedPoint = 3
        print("point 4")
    elif key is not -1:
        handleKeypress(key, selectedPoint)


def handleKeypress(key:int, selectedPoint:int):
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