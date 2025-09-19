import math
from typing import List, Tuple, Union, Dict
import numpy
import cv2.aruco as aruco
import numpy as np

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


def create_calibration_board(camera_config: Dict):
    """
    Create an ArUco board from the calibration marker configuration.
    This enables the use of refineDetectedMarkers for better detection.
    """
    markers = camera_config["calibration_markers"]
    measurements = camera_config["measurements"]
    
    # Extract marker IDs and their physical positions
    marker_ids = []
    marker_corners_3d = []
    
    for position, marker_info in markers.items():
        marker_id = int(marker_info["id"])
        physical_pos = marker_info["physical_position"]
        
        marker_ids.append(marker_id)
        
        # Create 4 corners for this marker in 3D space (Z=0 for planar board)
        # Assuming each marker is 1x1 unit square at its physical position
        corners_3d = np.array([
            [physical_pos[0] - 0.5, physical_pos[1] + 0.5, 0],  # top-left
            [physical_pos[0] + 0.5, physical_pos[1] + 0.5, 0],  # top-right  
            [physical_pos[0] + 0.5, physical_pos[1] - 0.5, 0],  # bottom-right
            [physical_pos[0] - 0.5, physical_pos[1] - 0.5, 0],  # bottom-left
        ], dtype=np.float32)
        
        marker_corners_3d.append(corners_3d)
    
    # Create the board
    board = aruco.Board(marker_corners_3d, aruco_dict, marker_ids)
    return board


def detect_markers_with_refinement(ir_image, camera_config: Dict) -> DetectionResult:
    """
    Detect markers with refinement based on board layout.
    """
    # Initial detection
    corners, ids, rejected = detect_markers(ir_image)
    initial_count = len(corners) if corners is not None else 0
    
    # Create board for this camera
    board = create_calibration_board(camera_config)
    
    # Refine detection using board layout
    detector = aruco.ArucoDetector(aruco_dict, parameters)
    
    # Use refineDetectedMarkers to find missing markers
    detector.refineDetectedMarkers(ir_image, board, corners, ids, rejected)
    
    refined_count = len(corners) if corners is not None else 0
    if refined_count > initial_count:
        print(f"🔍 Refinement found {refined_count - initial_count} additional markers!")
    
    return corners, ids, rejected


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
    