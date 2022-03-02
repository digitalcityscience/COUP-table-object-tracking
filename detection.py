from typing import Dict, List, Tuple
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


DetectionResult = Tuple[str, list[int], List]


def detect_markers(ir_image: List) -> DetectionResult:
    # corners, ids, rejectedImgPoints
    return aruco.detectMarkers(ir_image, aruco_dict, parameters=parameters)
