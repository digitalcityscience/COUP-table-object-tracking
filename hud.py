import math
from typing import Dict

import cv2
import cv2.aruco as aruco
import numpy

from marker import Marker

# Global storage for persistent marker positions
persistent_markers = {}

def update_persistent_markers(ids, corners):
    """Update the persistent marker storage with newly detected markers"""
    global persistent_markers
    
    if ids is not None and corners is not None:
        for i, marker_corners in enumerate(corners):
            if i < len(ids):
                marker_id = ids[i][0]
                # Calculate center of the marker
                center = numpy.mean(marker_corners[0], axis=0).astype(int)
                # Store the marker position
                persistent_markers[marker_id] = tuple(center)

def draw_persistent_marker_ids(ir_image, currently_detected_ids=None):
    """Draw all previously detected marker IDs at their last known positions"""
    global persistent_markers
    
    # Convert currently detected IDs to a set for fast lookup
    if currently_detected_ids is not None:
        current_set = set(currently_detected_ids.flatten()) if currently_detected_ids is not None else set()
    else:
        current_set = set()
    
    for marker_id, position in persistent_markers.items():
        # Only draw if not currently detected (to avoid duplicates)
        if marker_id not in current_set:
            cv2.putText(ir_image, str(marker_id), position, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


def draw_status_window(markerDict: Dict[int, Marker], camera_id: int = 0) -> None:
    status = numpy.zeros((800, 335, 3), numpy.uint8)
    statusY = 50
    linePositions = [15, 75, 130, 240, 290]
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(status, 'mId', (linePositions[0], statusY), font,  0.6, (255, 255, 255), 1)
    cv2.putText(status, 'xPos', (linePositions[1], statusY), font, 0.6, (255, 255, 255), 1)
    cv2.putText(
            status,
            f'rotDeg',
            (linePositions[2], statusY),
            font,
            0.6,
            (255, 255, 255),
            1,
        )
    cv2.putText(status, '?%', (linePositions[3], statusY), font, 0.6, (255, 255, 255), 1)
    cv2.putText(status, f'cId', (linePositions[4], statusY), font, 0.6, (255, 255, 255), 1)
    statusY += 35

    for x in markerDict:
        ctr = markerDict[x].getPos()[0]
        deg = markerDict[x].getPos()[2]
        id = markerDict[x].getID()
        conf = markerDict[x].getConfidence()

        cv2.putText(status, str(id), (linePositions[0], statusY), font, 0.8, (255, 255, 255), 1)
        cv2.putText(status, str(ctr), (linePositions[1], statusY), font, 0.6, (255, 255, 255), 1)
        cv2.putText(
            status,
            f'{deg:.2f}',
            (linePositions[2], statusY),
            font,
            0.6,
            (255, 255, 255),
            1,
        )
        cv2.putText(status, str(conf), (linePositions[3], statusY), font, 0.6, (255, 255, 255), 1)
        cv2.putText(status, f'{camera_id}', (linePositions[4], statusY), font, 0.6, (255, 255, 255), 1)
        statusY += 35
    
    window_name = f'Status_{camera_id}'
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(window_name, status)


def draw_monitor_window(ir_image, corners, rejectedImgPoints, camera_id: int = 0, ids=None) -> None:
    # convert image to COLOR_GRAY2BGR so that we can draw with color over it
    ir_image = cv2.cvtColor(ir_image, cv2.COLOR_GRAY2BGR)
    ir_image = aruco.drawDetectedMarkers(ir_image, corners, borderColor=(0, 255, 0))
    ir_image = aruco.drawDetectedMarkers(
        ir_image, rejectedImgPoints, borderColor=(0, 0, 255)
    )
    
    # Update persistent marker storage with newly detected markers
    update_persistent_markers(ids, corners)
    
    # Draw marker IDs if provided (currently detected markers in yellow)
    if ids is not None and corners is not None:
        for i, marker_corners in enumerate(corners):
            if i < len(ids):
                # Calculate center of the marker
                center = numpy.mean(marker_corners[0], axis=0).astype(int)
                # Draw the marker ID for currently detected markers
                cv2.putText(ir_image, str(ids[i][0]), tuple(center), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    # Draw persistent marker IDs (previously detected markers in cyan)
    draw_persistent_marker_ids(ir_image, ids)

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
    window_name = f'IR_{camera_id}'
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(window_name, ir_image)
    handle_key_presses()
    return ir_image


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
    elif key != -1:
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
