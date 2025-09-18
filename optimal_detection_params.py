#!/usr/bin/env python3
"""
Optimal ArUco Detection Parameters for 20x20 Pixel Markers

This script calculates and tests optimal parameters specifically 
for your 20x20 pixel markers based on theory and your test results.
"""

import cv2
import cv2.aruco as aruco
import numpy as np
import time
from typing import Dict

def calculate_optimal_parameters_20px() -> aruco.DetectorParameters:
    """
    Calculate optimal parameters for 20x20 pixel markers
    Based on computer vision theory and your test results
    """
    params = aruco.DetectorParameters()
    
    # === PERIMETER RATE PARAMETERS ===
    # For 20px markers: perimeter = 4 * 20 = 80 pixels
    # Assuming ~1500x800 stitched image: total perimeter ≈ 4600 pixels
    # Theoretical min rate: 80/4600 ≈ 0.017
    params.minMarkerPerimeterRate = 0.012  # 30% safety margin below theoretical
    params.maxMarkerPerimeterRate = 0.25   # Allow up to ~30px markers (some tolerance)
    
    # === ADAPTIVE THRESHOLD WINDOW SIZES ===
    # Key insight: Window should be smaller than marker but large enough for good thresholding
    # For 20px markers: optimal window ≈ 12-16 pixels (60-80% of marker size)
    params.adaptiveThreshWinSizeMin = 5    # Minimum useful window
    params.adaptiveThreshWinSizeMax = 17   # Sweet spot: smaller than marker, good thresholding
    params.adaptiveThreshWinSizeStep = 4   # Reasonable steps
    
    # === THRESHOLD SENSITIVITY ===
    # Balance between finding candidates and avoiding false positives
    params.adaptiveThreshConstant = 6      # Moderate - not too aggressive
    params.minOtsuStdDev = 2.0            # Moderate sensitivity
    
    # === CORNER DETECTION ACCURACY ===
    # Key insight: 20px markers need some tolerance but not too much
    params.polygonalApproxAccuracyRate = 0.04  # Balanced: not too strict, not too loose
    
    # === CORNER REFINEMENT ===
    # CORNER_REFINE_NONE is fastest, CONTOUR is most accurate
    params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX  # Good balance
    
    return params

def calculate_performance_optimized_20px() -> aruco.DetectorParameters:
    """
    Performance-optimized parameters for 20px markers
    Prioritizes speed while maintaining reasonable detection
    """
    params = aruco.DetectorParameters()
    
    # Tighter perimeter rates to reduce candidates
    params.minMarkerPerimeterRate = 0.015  # Closer to theoretical minimum
    params.maxMarkerPerimeterRate = 0.22   # Tighter upper bound
    
    # Smaller windows for speed
    params.adaptiveThreshWinSizeMin = 5
    params.adaptiveThreshWinSizeMax = 13   # Smaller than optimal but much faster
    params.adaptiveThreshWinSizeStep = 4
    
    # Less aggressive thresholding to reduce candidates
    params.adaptiveThreshConstant = 7      # Higher = fewer candidates
    params.minOtsuStdDev = 2.5            # Less sensitive
    
    # Moderate corner detection
    params.polygonalApproxAccuracyRate = 0.035
    
    # Fastest corner refinement
    params.cornerRefinementMethod = aruco.CORNER_REFINE_NONE
    
    return params

def calculate_accuracy_optimized_20px() -> aruco.DetectorParameters:
    """
    Accuracy-optimized parameters for 20px markers
    Prioritizes detection rate over speed
    """
    params = aruco.DetectorParameters()
    
    # Wider perimeter rates to catch more candidates
    params.minMarkerPerimeterRate = 0.008  # Lower to catch smaller perceived markers
    params.maxMarkerPerimeterRate = 0.3    # Higher to catch larger perceived markers
    
    # Optimal windows for 20px markers
    params.adaptiveThreshWinSizeMin = 5
    params.adaptiveThreshWinSizeMax = 19   # Optimal for 20px markers
    params.adaptiveThreshWinSizeStep = 4
    
    # More aggressive thresholding
    params.adaptiveThreshConstant = 4      # Lower = more candidates
    params.minOtsuStdDev = 1.5            # More sensitive
    
    # More tolerant corner detection
    params.polygonalApproxAccuracyRate = 0.06
    
    # Best corner refinement
    params.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR
    
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
    print("Optimal ArUco Parameters for 20x20 Pixel Markers")
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
    
    # Test all three parameter sets
    parameter_sets = [
        (calculate_optimal_parameters_20px(), "Balanced Optimal"),
        (calculate_performance_optimized_20px(), "Performance Optimized"),
        (calculate_accuracy_optimized_20px(), "Accuracy Optimized")
    ]
    
    results = []
    for params, name in parameter_sets:
        print(f"\nTesting {name}...")
        result = test_parameter_set(test_image, params, name)
        results.append(result)
        
        print(f"  Detected: {result['detected']:2d}/{result['total_candidates']:2d} "
              f"({result['detection_rate']*100:.1f}%) in {result['elapsed_time_ms']:.1f}ms")
    
    # Summary comparison
    print(f"\n" + "=" * 55)
    print("COMPARISON SUMMARY")
    print("=" * 55)
    print(f"{'Parameter Set':<20} {'Detected':<10} {'Rate':<8} {'Time(ms)':<10} {'Candidates':<10}")
    print("-" * 55)
    
    for result in results:
        print(f"{result['name']:<20} {result['detected']:2d}/{result['total_candidates']:<6} "
              f"{result['detection_rate']*100:5.1f}% {result['elapsed_time_ms']:8.1f} {result['total_candidates']:8d}")
    
    # Recommendations
    print(f"\n" + "=" * 55)
    print("RECOMMENDATIONS FOR 20x20 PIXEL MARKERS:")
    print("=" * 55)
    
    best_balanced = min(results, key=lambda x: abs(x['detection_rate'] - 0.5) + x['elapsed_time_ms']/1000)
    fastest = min(results, key=lambda x: x['elapsed_time_ms'])
    most_accurate = max(results, key=lambda x: x['detection_rate'])
    
    print(f"🎯 Best Balanced: {best_balanced['name']}")
    print(f"⚡ Fastest: {fastest['name']} ({fastest['elapsed_time_ms']:.1f}ms)")
    print(f"🔍 Most Accurate: {most_accurate['name']} ({most_accurate['detection_rate']*100:.1f}%)")
    
    print(f"\nKey Insights for 20px markers:")
    print(f"• Window size should be 13-19 pixels (65-95% of marker size)")
    print(f"• Perimeter rate: 0.012-0.025 (based on your image size)")
    print(f"• Polygon accuracy: 0.035-0.06 (balance speed vs accuracy)")
    print(f"• Avoid windows >20px - they're larger than your markers!")

if __name__ == "__main__":
    main() 