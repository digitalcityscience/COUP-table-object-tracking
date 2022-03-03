import math
from typing import List, Tuple, Union

import numpy
import cv2.aruco as aruco

aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
parameters = aruco.DetectorParameters_create()
parameters.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR
parameters.maxMarkerPerimeterRate = 0.2
parameters.minMarkerPerimeterRate = 0.05
parameters.polygonalApproxAccuracyRate = 0.03
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
    return aruco.detectMarkers(ir_image, aruco_dict, parameters=parameters)


def normalizeCorners(corner:Corner) -> List[Union[int, float]]:
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

    ctrX = numpy.interp(ctrX,[0,10000],[0,10000])
    ctrY = numpy.interp(ctrY,[0,10000],[0,10000])

    return [int(ctrX), int(ctrY), angleDeg]