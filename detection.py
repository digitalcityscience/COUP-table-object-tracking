import math
from typing import List, Tuple, Union

import numpy
import cv2.aruco as aruco

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
parameters = aruco.DetectorParameters()
parameters.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR
parameters.maxMarkerPerimeterRate = 0.3  # increased from 0.2
parameters.minMarkerPerimeterRate = 0.03  # decreased from 0.05
parameters.polygonalApproxAccuracyRate = 0.05  # increased from 0.02
parameters.minOtsuStdDev = 5.0  # uncommented and set
# parameters.perspectiveRemovePixelPerCell = 10
# parameters.perspectiveRemoveIgnoredMarginPerCell = 0.13
# parameters.errorCorrectionRate = 0.3

parameters.adaptiveThreshWinSizeMin = 3  # uncommented
parameters.adaptiveThreshWinSizeMax = 23  # uncommented
parameters.adaptiveThreshWinSizeStep = 10  # uncommented
parameters.adaptiveThreshConstant = 7  # uncommented

Corner = numpy.ndarray
DetectionResult = Tuple[List[Corner], List[int], List]

def detect_markers(ir_image: List) -> DetectionResult:
    # corners, ids, rejectedImgPoints
    return aruco.detectMarkers(ir_image, aruco_dict, parameters=parameters)


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
    