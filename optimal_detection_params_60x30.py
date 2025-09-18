#!/usr/bin/env python3
"""
Optimal ArUco Detection Parameters for 60x30 Pixel Markers

This script calculates optimal parameters specifically for rectangular 
60x30 pixel markers, accounting for their larger size and aspect ratio.
"""

import cv2
import cv2.aruco as aruco
import numpy as np
import time
from typing import Dict

def calculate_optimal_parameters_60x30() -> aruco.DetectorParameters:
    """
    Calculate optimal parameters for 60x30 pixel markers
    Based on computer vision theory for rectangular markers
    """
    params = aruco.DetectorParameters()
    
    # === PERIMETER RATE PARAMETERS ===
    # For 60x30px markers: perimeter = 2*(60+30) = 180 pixels
    # Assuming ~1500x800 stitched image: total perimeter ≈ 4600 pixels
    # Theoretical min rate: 180/4600 ≈ 0.039
    params.minMarkerPerimeterRate = 0.025  # 35% safety margin below theoretical
    params.maxMarkerPerimeterRate = 0.15   # Allow up to ~80x40px markers (some tolerance)
    
    # === ADAPTIVE THRESHOLD WINDOW SIZES ===
    # Key insight: Window should be smaller than shortest dimension but large enough for good thresholding
    # For 60x30px markers: shortest dimension is 30px
    # Optimal window ≈ 18-25 pixels (60-85% of shortest dimension)
    params.adaptiveThreshWinSizeMin = 7    # Minimum useful window for larger markers
    params.adaptiveThreshWinSizeMax = 25   # Sweet spot: smaller than shortest dimension
    params.adaptiveThreshWinSizeStep = 6   # Reasonable steps for larger range
    
    # === THRESHOLD SENSITIVITY ===
    # Larger markers can tolerate more aggressive thresholding
    params.adaptiveThreshConstant = 7      # Moderate - larger markers are more robust
    params.minOtsuStdDev = 3.0            # Moderate sensitivity for larger features
    
    # === CORNER DETECTION ACCURACY ===
    # Larger markers can be more precise in corner detection
    params.polygonalApproxAccuracyRate = 0.03  # More precise than small markers
    
    # === CORNER REFINEMENT ===
    # Larger markers benefit from better corner refinement
    params.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR  # Best accuracy
    
    return params

def calculate_performance_optimized_60x30() -> aruco.DetectorParameters:
    """
    Performance-optimized parameters for 60x30px markers
    Prioritizes speed while maintaining reasonable detection
    """
    params = aruco.DetectorParameters()
    
    # Tighter perimeter rates to reduce candidates
    params.minMarkerPerimeterRate = 0.03   # Closer to theoretical minimum
    params.maxMarkerPerimeterRate = 0.12   # Tighter upper bound
    
    # Smaller windows for speed (but still appropriate for 60x30)
    params.adaptiveThreshWinSizeMin = 7
    params.adaptiveThreshWinSizeMax = 21   # Smaller than optimal but much faster
    params.adaptiveThreshWinSizeStep = 6
    
    # Less aggressive thresholding to reduce candidates
    params.adaptiveThreshConstant = 8      # Higher = fewer candidates
    params.minOtsuStdDev = 3.5            # Less sensitive
    
    # Moderate corner detection
    params.polygonalApproxAccuracyRate = 0.035
    
    # Fastest corner refinement
    params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX  # Good balance
    
    return params

def calculate_accuracy_optimized_60x30() -> aruco.DetectorParameters:
    """
    Accuracy-optimized parameters for 60x30px markers
    Prioritizes detection rate over speed
    """
    params = aruco.DetectorParameters()
    
    # Wider perimeter rates to catch more candidates
    params.minMarkerPerimeterRate = 0.02   # Lower to catch smaller perceived markers
    params.maxMarkerPerimeterRate = 0.18   # Higher to catch larger perceived markers
    
    # Optimal windows for 60x30px markers
    params.adaptiveThreshWinSizeMin = 7
    params.adaptiveThreshWinSizeMax = 27   # Optimal for 30px shortest dimension
    params.adaptiveThreshWinSizeStep = 6
    
    # More aggressive thresholding
    params.adaptiveThreshConstant = 5      # Lower = more candidates
    params.minOtsuStdDev = 2.5            # More sensitive
    
    # More tolerant corner detection for edge cases
    params.polygonalApproxAccuracyRate = 0.04
    
    # Best corner refinement
    params.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR
    
    return params

def calculate_rectangular_marker_params(width_px: int, height_px: int, image_width: int = 1500, image_height: int = 800) -> aruco.DetectorParameters:
    """
    Generic function to calculate optimal parameters for any rectangular marker size
    
    Args:
        width_px: Marker width in pixels
        height_px: Marker height in pixels  
        image_width: Stitched image width
        image_height: Stitched image height
    """
    params = aruco.DetectorParameters()
    
    # Calculate marker properties
    marker_perimeter = 2 * (width_px + height_px)
    image_perimeter = 2 * (image_width + image_height)
    shortest_dimension = min(width_px, height_px)
    
    # Perimeter rates based on actual dimensions
    theoretical_min_rate = marker_perimeter / image_perimeter
    params.minMarkerPerimeterRate = theoretical_min_rate * 0.7  # 30% safety margin
    params.maxMarkerPerimeterRate = theoretical_min_rate * 2.5  # Allow 2.5x size variation
    
    # Adaptive window sizes based on shortest dimension
    optimal_window = int(shortest_dimension * 0.75)  # 75% of shortest dimension
    params.adaptiveThreshWinSizeMin = max(3, optimal_window // 3)
    params.adaptiveThreshWinSizeMax = min(31, optimal_window)  # Cap at OpenCV limit
    params.adaptiveThreshWinSizeStep = max(2, (params.adaptiveThreshWinSizeMax - params.adaptiveThreshWinSizeMin) // 3)
    
    # Threshold parameters based on marker size
    if shortest_dimension < 20:
        # Small markers - more sensitive
        params.adaptiveThreshConstant = 5
        params.minOtsuStdDev = 2.0
        params.polygonalApproxAccuracyRate = 0.05
    elif shortest_dimension < 40:
        # Medium markers - balanced
        params.adaptiveThreshConstant = 6
        params.minOtsuStdDev = 2.5
        params.polygonalApproxAccuracyRate = 0.04
    else:
        # Large markers - can be more precise
        params.adaptiveThreshConstant = 7
        params.minOtsuStdDev = 3.0
        params.polygonalApproxAccuracyRate = 0.03
    
    # Corner refinement based on marker size
    if shortest_dimension >= 30:
        params.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR  # Best for large markers
    elif shortest_dimension >= 15:
        params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX   # Balanced
    else:
        params.cornerRefinementMethod = aruco.CORNER_REFINE_NONE     # Fastest for small markers
    
    return params

def test_parameter_set(test_image: np.ndarray, params: aruco.DetectorParameters, name: str) -> Dict:
    """Test a parameter set and return performance metrics"""
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
    
    # Time the detection
    start_time = time.time()
    corners, ids, rejected = aruco.detectMarkers(test_image, aruco_dict, parameters=params)
    end_time = time.time()
    
    detected_count = len(ids) if ids is not None else 0
    rejected_count = len(rejected) if rejected is not None else 0
    total_candidates = detected_count + rejected_count
    detection_rate = detected_count / total_candidates if total_candidates > 0 else 0
    elapsed_time = (end_time - start_time) * 1000  # Convert to milliseconds
    
    result = {
        'name': name,
        'detected': detected_count,
        'rejected': rejected_count,
        'total_candidates': total_candidates,
        'detection_rate': detection_rate,
        'elapsed_time_ms': elapsed_time,
        'parameters': {
            'minMarkerPerimeterRate': params.minMarkerPerimeterRate,
            'maxMarkerPerimeterRate': params.maxMarkerPerimeterRate,
            'polygonalApproxAccuracyRate': params.polygonalApproxAccuracyRate,
            'minOtsuStdDev': params.minOtsuStdDev,
            'adaptiveThreshWinSizeMin': params.adaptiveThreshWinSizeMin,
            'adaptiveThreshWinSizeMax': params.adaptiveThreshWinSizeMax,
            'adaptiveThreshWinSizeStep': params.adaptiveThreshWinSizeStep,
            'adaptiveThreshConstant': params.adaptiveThreshConstant,
            'cornerRefinementMethod': params.cornerRefinementMethod
        }
    }
    
    return result

def main():
    print("Optimal ArUco Parameters for 60x30 Pixel Markers")
    print("=" * 55)
    
    # Load test image
    import os
    test_files = [f for f in os.listdir('.') if f.startswith('test_stitched_image_') and f.endswith('.png')]
    if not test_files:
        print("No test image found! Run the main test script first.")
        return
    
    latest_file = sorted(test_files)[-1]
    print(f"Using test image: {latest_file}")
    test_image = cv2.imread(latest_file, cv2.IMREAD_GRAYSCALE)
    
    # Test all parameter sets for 60x30 markers
    parameter_sets = [
        (calculate_optimal_parameters_60x30(), "Balanced Optimal (60x30)"),
        (calculate_performance_optimized_60x30(), "Performance Optimized (60x30)"),
        (calculate_accuracy_optimized_60x30(), "Accuracy Optimized (60x30)"),
        (calculate_rectangular_marker_params(60, 30), "Generic Algorithm (60x30)")
    ]
    
    results = []
    for params, name in parameter_sets:
        print(f"\nTesting {name}...")
        result = test_parameter_set(test_image, params, name)
        results.append(result)
        
        print(f"  Detected: {result['detected']:2d}/{result['total_candidates']:2d} "
              f"({result['detection_rate']*100:.1f}%) in {result['elapsed_time_ms']:.1f}ms")
        
        # Show key parameters
        p = result['parameters']
        print(f"  Key params: perimeter({p['minMarkerPerimeterRate']:.3f}-{p['maxMarkerPerimeterRate']:.3f}), "
              f"window({p['adaptiveThreshWinSizeMin']}-{p['adaptiveThreshWinSizeMax']}), "
              f"poly({p['polygonalApproxAccuracyRate']:.3f})")
    
    # Summary comparison
    print(f"\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Parameter Set':<25} {'Detected':<10} {'Rate':<8} {'Time(ms)':<10} {'Candidates':<10}")
    print("-" * 70)
    
    for result in results:
        print(f"{result['name']:<25} {result['detected']:2d}/{result['total_candidates']:<6} "
              f"{result['detection_rate']*100:5.1f}% {result['elapsed_time_ms']:8.1f} {result['total_candidates']:8d}")
    
    # Recommendations
    print(f"\n" + "=" * 70)
    print("RECOMMENDATIONS FOR 60x30 PIXEL MARKERS:")
    print("=" * 70)
    
    best_balanced = min(results, key=lambda x: abs(x['detection_rate'] - 0.7) + x['elapsed_time_ms']/1000)
    fastest = min(results, key=lambda x: x['elapsed_time_ms'])
    most_accurate = max(results, key=lambda x: x['detection_rate'])
    
    print(f"🎯 Best Balanced: {best_balanced['name']}")
    print(f"⚡ Fastest: {fastest['name']} ({fastest['elapsed_time_ms']:.1f}ms)")
    print(f"🔍 Most Accurate: {most_accurate['name']} ({most_accurate['detection_rate']*100:.1f}%)")
    
    print(f"\nKey Insights for 60x30px markers:")
    print(f"• Perimeter: 180 pixels (vs 80 for 20x20) - much larger search space")
    print(f"• Window size should be 21-27 pixels (70-90% of shortest dimension: 30px)")
    print(f"• Perimeter rate: 0.025-0.15 (based on larger marker size)")
    print(f"• Can use more precise corner detection due to larger size")
    print(f"• Rectangular shape may cause some detection challenges vs squares")
    
    # Show optimal parameters for copy-paste
    best_params = best_balanced['parameters']
    print(f"\n" + "=" * 70)
    print("COPY-PASTE OPTIMAL PARAMETERS:")
    print("=" * 70)
    print("params.minMarkerPerimeterRate = {:.3f}".format(best_params['minMarkerPerimeterRate']))
    print("params.maxMarkerPerimeterRate = {:.3f}".format(best_params['maxMarkerPerimeterRate']))
    print("params.adaptiveThreshWinSizeMin = {}".format(best_params['adaptiveThreshWinSizeMin']))
    print("params.adaptiveThreshWinSizeMax = {}".format(best_params['adaptiveThreshWinSizeMax']))
    print("params.adaptiveThreshWinSizeStep = {}".format(best_params['adaptiveThreshWinSizeStep']))
    print("params.polygonalApproxAccuracyRate = {:.3f}".format(best_params['polygonalApproxAccuracyRate']))
    print("params.minOtsuStdDev = {:.1f}".format(best_params['minOtsuStdDev']))
    print("params.adaptiveThreshConstant = {:.0f}".format(best_params['adaptiveThreshConstant']))
    print("params.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR")

if __name__ == "__main__":
    main() 