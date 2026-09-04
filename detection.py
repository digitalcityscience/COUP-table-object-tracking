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

# OpenCV 5+ exposes marker detection via ArucoDetector, while older
# versions expose the module function aruco.detectMarkers.
detector = aruco.ArucoDetector(aruco_dict, parameters) if hasattr(aruco, "ArucoDetector") else None

Corner = numpy.ndarray
DetectionResult = Tuple[List[Corner], List[int], List]

def detect_markers(ir_image: List) -> DetectionResult:
    # corners, ids, rejectedImgPoints
    if detector is not None:
        return detector.detectMarkers(ir_image)
    return aruco.detectMarkers(ir_image, aruco_dict, parameters=parameters)


def normalizeCorners(coords:Corner) -> Tuple[int,int,float]:

    p1 = tuple(coords[0][0])
    p3 = tuple(coords[0][2])

    centerX = (p1[0] + p3[0]) / 2
    centerY = (p1[1] + p3[1]) / 2

    dx = p1[0] - centerX
    dy = p1[1] - centerY

    angle = math.atan2(dy,dx)
    # The `-1` that used to sit here dated from a single mirrored camera feed. Stitching now
    # hands us a table frame with +x = East and +y = North (det(H) > 0, no mirroring), so the
    # negation stopped being a correction and became the bug: turning a block +90 deg on the
    # table rotated its footprint -90 deg on the map. Measured on the rig 2026-09-04 across
    # four orientations of G11 -- the map tracked the block at exactly -1x, within 1.3 deg.
    angleDeg = math.degrees(angle)
    #angleDeg = (angleDeg + 360) % 360  # map from -180<->180 to 0<->360

    centerX = numpy.interp(centerX,[0,10000],[0,10000])
    centerY = numpy.interp(centerY,[0,10000],[0,10000])

    return int(centerX), int(centerY), angleDeg
    
