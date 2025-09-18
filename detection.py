import math
from typing import List, Tuple, Union

import numpy
import cv2.aruco as aruco

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)

# Parameters for individual camera streams (original)
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

def calculate_stitched_parameters(scale_factor: float, marker_size_cm: float = 2.0) -> aruco.DetectorParameters:
    """
    Calculate optimal ArUco detection parameters for stitched images
    
    Args:
        scale_factor: Pixels per cm in the stitched image
        marker_size_cm: Physical size of markers in centimeters
    
    Returns:
        Optimized DetectorParameters for the given scale
    """
    # Calculate marker size in pixels
    marker_size_pixels = marker_size_cm * scale_factor
    
    # Estimate typical stitched image dimensions (rough approximation)
    # This affects perimeter rate calculations
    typical_stitched_width = 1500  # pixels (approximate)
    typical_stitched_height = 800   # pixels (approximate)
    typical_perimeter = 2 * (typical_stitched_width + typical_stitched_height)
    
    params = aruco.DetectorParameters()
    params.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR
    
    # === PERIMETER RATE PARAMETERS ===
    # Calculate based on actual marker size vs image size
    marker_perimeter = 4 * marker_size_pixels
    base_min_perimeter_rate = marker_perimeter / typical_perimeter
    
    # Set with safety margins
    params.minMarkerPerimeterRate = max(0.005, base_min_perimeter_rate * 0.7)  # 30% safety margin
    params.maxMarkerPerimeterRate = min(0.5, base_min_perimeter_rate * 3.0)    # allow 3x larger
    
    # === ADAPTIVE THRESHOLD WINDOW SIZES ===
    # Base window size on marker size - should be smaller than marker
    optimal_window_size = int(marker_size_pixels * 0.6)  # 60% of marker size
    
    params.adaptiveThreshWinSizeMin = max(3, optimal_window_size // 3)
    params.adaptiveThreshWinSizeMax = max(7, min(optimal_window_size, 25))  # cap at 25
    params.adaptiveThreshWinSizeStep = max(2, (params.adaptiveThreshWinSizeMax - params.adaptiveThreshWinSizeMin) // 3)
    
    # === ACCURACY PARAMETERS ===
    # Smaller markers need more precision
    if marker_size_pixels < 25:
        params.polygonalApproxAccuracyRate = 0.02  # high precision
        params.minOtsuStdDev = 2.0  # very sensitive
        params.adaptiveThreshConstant = 4
    elif marker_size_pixels < 40:
        params.polygonalApproxAccuracyRate = 0.03  # medium precision
        params.minOtsuStdDev = 3.0  # medium sensitive
        params.adaptiveThreshConstant = 5
    else:
        params.polygonalApproxAccuracyRate = 0.04  # standard precision
        params.minOtsuStdDev = 4.0  # less sensitive
        params.adaptiveThreshConstant = 6
    
    print(f"Calculated stitched detection parameters for {marker_size_cm}cm markers at {scale_factor} px/cm:")
    print(f"  Marker size: {marker_size_pixels:.1f} pixels")
    print(f"  Min perimeter rate: {params.minMarkerPerimeterRate:.4f}")
    print(f"  Max perimeter rate: {params.maxMarkerPerimeterRate:.4f}")
    print(f"  Adaptive window: {params.adaptiveThreshWinSizeMin}-{params.adaptiveThreshWinSizeMax} (step {params.adaptiveThreshWinSizeStep})")
    print(f"  Polygon accuracy: {params.polygonalApproxAccuracyRate:.3f}")
    
    return params

# Global variable to store calculated stitched parameters
_stitched_parameters = None

def set_stitched_detection_parameters(scale_factor: float, marker_size_cm: float = 2.0):
    """
    Set the parameters for stitched image detection based on scale factor
    Call this once during setup with your stitching configuration
    """
    global _stitched_parameters
    _stitched_parameters = calculate_stitched_parameters(scale_factor, marker_size_cm)

Corner = numpy.ndarray
DetectionResult = Tuple[List[Corner], List[int], List]

def detect_markers(ir_image: List) -> DetectionResult:
    # corners, ids, rejectedImgPoints
    return aruco.detectMarkers(ir_image, aruco_dict, parameters=parameters)

def detect_markers_stitched(ir_image: List) -> DetectionResult:
    """
    Detect markers optimized for stitched images
    Parameters are automatically calculated based on the scale factor
    Call set_stitched_detection_parameters() first to configure
    """
    global _stitched_parameters
    if _stitched_parameters is None:
        raise RuntimeError("Stitched detection parameters not set. Call set_stitched_detection_parameters() first.")
    
    return aruco.detectMarkers(ir_image, aruco_dict, parameters=_stitched_parameters)


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
    