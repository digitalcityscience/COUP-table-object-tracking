import math
from typing import List, Tuple, Union

import numpy
import cv2.aruco as aruco
import cv2 as cv

aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_250)
parameters =  cv.aruco.DetectorParameters()
parameters.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR
parameters.maxMarkerPerimeterRate = 0.2
parameters.minMarkerPerimeterRate = 0.05
parameters.polygonalApproxAccuracyRate = 0.03
detector = cv.aruco.ArucoDetector(aruco_dict, parameters)

# parameters.minOtsuStdDev = 2.0
# parameters.perspectiveRemovePixelPerCell = 10
# parameters.perspectiveRemoveIgnoredMarginPerCell = 0.13
# parameters.errorCorrectionRate = 0.3

# parameters.adaptiveThreshWinSizeMin = 3
# parameters.adaptiveThreshWinSizeMax = 23
# parameters.adaptiveThreshWinSizeStep = 5
# parameters.adaptiveThreshConstant = 7

Corner = numpy.ndarray
DetectionResult = Tuple[List[Corner], List[int], List]

def detect_markers(ir_image: List) -> DetectionResult:
    # corners, ids, rejectedImgPoints
    print(type(detector.detectMarkers(ir_image)))
    print(detector.detectMarkers(ir_image))
    return detector.detectMarkers(ir_image)


def normalizeCorners(coords:Corner) -> Tuple[int,int,float]:

    p1 = tuple(coords[0][0])
    p3 = tuple(coords[0][2])

    centerX = (p1[0] + p3[0]) / 2
    centerY = (p1[1] + p3[1]) / 2

    dx = p1[0] - centerX
    dy = p1[1] - centerY

    angle = math.atan2(dy,dx)
    angleDeg = math.degrees(angle)
    mirroredAngleDeg = -angleDeg #we need do multiply with -1 because the picture that we got is mirrord
    #angleDeg = (angleDeg + 360) % 360  # map from -180<->180 to 0<->360

    centerX = numpy.interp(centerX,[0,10000],[0,10000])
    centerY = numpy.interp(centerY,[0,10000],[0,10000])
    
    return int(centerX), int(centerY), mirroredAngleDeg
    