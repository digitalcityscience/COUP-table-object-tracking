#!/usr/bin/env python3
"""
ArUco Detection Parameter Optimization Script

This script tests all permutations of ArUco detection parameters
to find the optimal settings for your stitched images.
"""

import cv2
import cv2.aruco as aruco
import numpy as np
import json
import os
from typing import Dict, List, Tuple
import itertools
from datetime import datetime

# Import your existing functions
from camera_stitching import setup_camera_transforms, process_and_join_streams
from calibration_handler import load_calibration_markers

def test_aruco_parameters_comprehensive(test_image: np.ndarray, save_results: bool = True) -> List[Dict]:
    """
    Test all permutations of ArUco detection parameters
    
    Args:
        test_image: Image to test detection on
        save_results: Whether to save results to JSON file
    
    Returns:
        List of dictionaries with parameter combinations and results
    """
    
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
    results = []
    
    # Define parameter ranges to test - OPTIMIZED for 30x30px markers on 2400x1200 image
    # Marker perimeter ~120px, image perimeter ~7200px, ratio ~0.0167
    param_ranges = {
        'minMarkerPerimeterRate': [0.008, 0.010, 0.012, 0.015, 0.018, 0.020, 0.025],  # Focused around 0.0167
        'maxMarkerPerimeterRate': [0.15, 0.2, 0.25, 0.3, 0.4, 0.5],  # Slightly lower max values
        'polygonalApproxAccuracyRate': [0.015, 0.02, 0.025, 0.03, 0.035, 0.04],  # Tighter range for small markers
        'minOtsuStdDev': [2.0, 3.0, 4.0, 5.0, 6.0],  # Good range for marker contrast
        'adaptiveThreshWinSizeMin': [3, 5, 7, 9],  # Start with smaller windows
        'adaptiveThreshWinSizeMax': [15, 19, 23, 31, 39],  # Appropriate for 30px markers
        'adaptiveThreshWinSizeStep': [2, 4, 6, 8],  # Good step sizes
        'adaptiveThreshConstant': [5, 7, 9, 11],  # Focused range
        'cornerRefinementMethod': [
            aruco.CORNER_REFINE_NONE,
            aruco.CORNER_REFINE_SUBPIX,
            aruco.CORNER_REFINE_CONTOUR
        ]
    }
    
    # Calculate total combinations
    total_combinations = 1
    for param_values in param_ranges.values():
        total_combinations *= len(param_values)
    
    print(f"Testing {total_combinations:,} parameter combinations...")
    print("This may take a while - progress will be shown every 100 tests")
    print("=" * 60)
    
    test_count = 0
    best_result = {'detected': 0, 'params': {}}
    
    # Test all permutations
    for combination in itertools.product(*param_ranges.values()):
        test_count += 1
        
        # Create parameter object
        params = aruco.DetectorParameters()
        param_names = list(param_ranges.keys())
        
        # Set parameters
        params.minMarkerPerimeterRate = combination[0]
        params.maxMarkerPerimeterRate = combination[1] 
        params.polygonalApproxAccuracyRate = combination[2]
        params.minOtsuStdDev = combination[3]
        params.adaptiveThreshWinSizeMin = combination[4]
        params.adaptiveThreshWinSizeMax = combination[5]
        params.adaptiveThreshWinSizeStep = combination[6]
        params.adaptiveThreshConstant = combination[7]
        params.cornerRefinementMethod = combination[8]
        
        # Skip invalid combinations
        if params.adaptiveThreshWinSizeMax <= params.adaptiveThreshWinSizeMin:
            continue
        if params.maxMarkerPerimeterRate <= params.minMarkerPerimeterRate:
            continue
            
        # Test detection
        try:
            corners, ids, rejected = aruco.detectMarkers(test_image, aruco_dict, parameters=params)
            
            detected_count = len(ids) if ids is not None else 0
            rejected_count = len(rejected) if rejected is not None else 0
            
            # Store result
            result = {
                'test_id': test_count,
                'detected': detected_count,
                'rejected': rejected_count,
                'total_candidates': detected_count + rejected_count,
                'detection_rate': detected_count / (detected_count + rejected_count) if (detected_count + rejected_count) > 0 else 0,
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
            results.append(result)
            
            # Track best result
            if detected_count > best_result['detected']:
                best_result = {
                    'detected': detected_count,
                    'rejected': rejected_count,
                    'params': result['parameters'].copy()
                }
            
            # Progress update
            if test_count % 100 == 0:
                print(f"Progress: {test_count:,}/{total_combinations:,} "
                      f"({100*test_count/total_combinations:.1f}%) - "
                      f"Best so far: {best_result['detected']} detected")
                
        except Exception as e:
            print(f"Error with combination {test_count}: {e}")
            continue
    
    print("\n" + "=" * 60)
    print(f"Testing complete! Tested {len(results)} valid combinations")
    print(f"Best result: {best_result['detected']} markers detected, {best_result.get('rejected', 0)} rejected")
    
    # Sort results by detection count (descending)
    results.sort(key=lambda x: (x['detected'], -x['rejected']), reverse=True)
    
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"aruco_parameter_test_results_{timestamp}.json"
        
        # Save detailed results
        with open(filename, 'w') as f:
            json.dump({
                'test_summary': {
                    'total_tests': len(results),
                    'best_detected': best_result['detected'],
                    'best_parameters': best_result['params'],
                    'timestamp': timestamp
                },
                'results': results[:50]  # Save top 50 results
            }, f, indent=2)
        
        print(f"Results saved to: {filename}")
    
    return results

def test_quick_parameter_sweep(test_image: np.ndarray) -> List[Dict]:
    """
    Quick test of most important parameters for troubleshooting
    """
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
    results = []
    
    # Focus on the most critical parameters - OPTIMIZED for 30x30px markers
    min_perimeter_rates = [0.008, 0.010, 0.012, 0.015, 0.018, 0.020, 0.025]  # Centered around theoretical 0.0167
    max_perimeter_rates = [0.15, 0.2, 0.25, 0.3, 0.4]  # More focused range
    window_max_sizes = [15, 19, 23, 31, 39]  # Better for 30px markers
    
    print(f"Quick parameter sweep: {len(min_perimeter_rates) * len(max_perimeter_rates) * len(window_max_sizes)} combinations")
    
    test_count = 0
    for min_rate in min_perimeter_rates:
        for max_rate in max_perimeter_rates:
            for win_max in window_max_sizes:
                if max_rate <= min_rate:
                    continue
                    
                test_count += 1
                
                # Create parameters
                params = aruco.DetectorParameters()
                params.minMarkerPerimeterRate = min_rate
                params.maxMarkerPerimeterRate = max_rate
                params.adaptiveThreshWinSizeMin = 3
                params.adaptiveThreshWinSizeMax = win_max
                params.adaptiveThreshWinSizeStep = min(6, (win_max - 3) // 2)
                params.adaptiveThreshConstant = 7
                params.polygonalApproxAccuracyRate = 0.03
                params.minOtsuStdDev = 3.0
                params.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR
                
                # Test
                corners, ids, rejected = aruco.detectMarkers(test_image, aruco_dict, parameters=params)
                
                detected_count = len(ids) if ids is not None else 0
                rejected_count = len(rejected) if rejected is not None else 0
                
                result = {
                    'detected': detected_count,
                    'rejected': rejected_count,
                    'minPerimeterRate': min_rate,
                    'maxPerimeterRate': max_rate,
                    'windowMax': win_max
                }
                results.append(result)
                
                print(f"Test {test_count:3d}: min={min_rate:.3f}, max={max_rate:.1f}, win={win_max:2d} -> "
                      f"detected={detected_count}, rejected={rejected_count}")
    
    # Sort by detected count
    results.sort(key=lambda x: x['detected'], reverse=True)
    return results

def test_parameters_for_30px_markers(test_image: np.ndarray) -> List[Dict]:
    """
    Highly targeted parameter test for 30x30 pixel markers on 2400x1200 images
    
    Based on theoretical calculations:
    - Marker perimeter: 30*4 = 120 pixels
    - Image perimeter: (2400+1200)*2 = 7200 pixels
    - Theoretical perimeter rate: 120/7200 = 0.0167
    """
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
    results = []
    
    # Highly focused parameters for 30px markers
    min_perimeter_rates = [0.012, 0.015, 0.017, 0.020]  # Very tight around 0.0167
    max_perimeter_rates = [0.2, 0.25, 0.3]  # Conservative upper bounds
    window_sizes = [(3, 15), (3, 19), (5, 23), (7, 31)]  # (min, max) pairs appropriate for 30px
    constants = [7, 9]  # Good middle values
    poly_accuracies = [0.02, 0.025, 0.03]  # Tight for small markers
    
    total_tests = len(min_perimeter_rates) * len(max_perimeter_rates) * len(window_sizes) * len(constants) * len(poly_accuracies)
    print(f"Targeted test for 30px markers: {total_tests} combinations")
    
    test_count = 0
    for min_rate in min_perimeter_rates:
        for max_rate in max_perimeter_rates:
            for win_min, win_max in window_sizes:
                for constant in constants:
                    for poly_acc in poly_accuracies:
                        test_count += 1
                        
                        # Create optimized parameters
                        params = aruco.DetectorParameters()
                        params.minMarkerPerimeterRate = min_rate
                        params.maxMarkerPerimeterRate = max_rate
                        params.adaptiveThreshWinSizeMin = win_min
                        params.adaptiveThreshWinSizeMax = win_max
                        params.adaptiveThreshWinSizeStep = min(6, (win_max - win_min) // 2)
                        params.adaptiveThreshConstant = constant
                        params.polygonalApproxAccuracyRate = poly_acc
                        params.minOtsuStdDev = 3.0  # Good default
                        params.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR
                        
                        # Test detection
                        corners, ids, rejected = aruco.detectMarkers(test_image, aruco_dict, parameters=params)
                        
                        detected_count = len(ids) if ids is not None else 0
                        rejected_count = len(rejected) if rejected is not None else 0
                        
                        result = {
                            'detected': detected_count,
                            'rejected': rejected_count,
                            'minPerimeterRate': min_rate,
                            'maxPerimeterRate': max_rate,
                            'windowMin': win_min,
                            'windowMax': win_max,
                            'constant': constant,
                            'polyAccuracy': poly_acc,
                            'detection_ratio': detected_count / (detected_count + rejected_count) if (detected_count + rejected_count) > 0 else 0
                        }
                        results.append(result)
                        
                        print(f"Test {test_count:3d}: min={min_rate:.3f}, max={max_rate:.2f}, "
                              f"win={win_min}-{win_max}, const={constant}, poly={poly_acc:.3f} -> "
                              f"detected={detected_count}, rejected={rejected_count}")
    
    # Sort by detected count, then by detection ratio
    results.sort(key=lambda x: (x['detected'], x['detection_ratio']), reverse=True)
    return results

def capture_test_image_from_stream():
    """
    Capture a single stitched image for testing
    """
    print("Capturing test image from camera stream...")
    
    # Load calibration and setup stitching
    calibration_data = load_calibration_markers("calibration_markers.json")
    setup_config = setup_camera_transforms(calibration_data)
    
    # Capture one stitched frame
    for stitched_image in process_and_join_streams(setup_config):
        print(f"Captured test image: {stitched_image.shape}")
        
        # Save the test image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_image_path = f"test_stitched_image_{timestamp}.png"
        cv2.imwrite(test_image_path, stitched_image)
        print(f"Test image saved as: {test_image_path}")
        
        return stitched_image, test_image_path

def main():
    print("ArUco Detection Parameter Optimization")
    print("=" * 40)
    
    # Choose test mode
    print("1. Capture new test image from camera stream")
    print("2. Use existing image file")
    print("3. Quick parameter sweep (recommended first)")
    print("4. Targeted test for 30x30px markers (RECOMMENDED for your setup)")
    print("5. Full comprehensive test (very slow)")
    
    choice = input("Choose option (1-5): ").strip()
    
    test_image = None
    
    if choice == "1":
        test_image, image_path = capture_test_image_from_stream()
        
    elif choice == "2":
        image_path = input("Enter path to test image: ").strip()
        if not os.path.exists(image_path):
            print(f"File not found: {image_path}")
            return
        test_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
    elif choice in ["3", "4", "5"]:
        # Try to find a recent test image
        test_files = [f for f in os.listdir('.') if f.startswith('test_stitched_image_') and f.endswith('.png')]
        if test_files:
            latest_file = sorted(test_files)[-1]
            print(f"Using latest test image: {latest_file}")
            test_image = cv2.imread(latest_file, cv2.IMREAD_GRAYSCALE)
        else:
            print("No test image found. Capturing from stream...")
            test_image, _ = capture_test_image_from_stream()
    
    if test_image is None:
        print("Failed to load test image")
        return
        
    print(f"Test image loaded: {test_image.shape}")
    
    # Run the selected test
    if choice == "3":
        results = test_quick_parameter_sweep(test_image)
        print(f"\nTop 10 Results:")
        print("-" * 60)
        for i, result in enumerate(results[:10]):
            print(f"{i+1:2d}. Detected: {result['detected']:2d}, Rejected: {result['rejected']:2d} "
                  f"(min={result['minPerimeterRate']:.3f}, max={result['maxPerimeterRate']:.1f}, "
                  f"win={result['windowMax']:2d})")
                  
    elif choice == "4":
        results = test_parameters_for_30px_markers(test_image)
        print(f"\nTop 10 Results for 30x30px markers:")
        print("-" * 80)
        for i, result in enumerate(results[:10]):
            print(f"{i+1:2d}. Detected: {result['detected']:2d}, Rejected: {result['rejected']:2d}, "
                  f"Ratio: {result['detection_ratio']:.2f}")
            print(f"    min={result['minPerimeterRate']:.3f}, max={result['maxPerimeterRate']:.2f}, "
                  f"win={result['windowMin']}-{result['windowMax']}")
            print(f"    const={result['constant']}, polyAcc={result['polyAccuracy']:.3f}")
            print()
                  
    elif choice == "5":
        results = test_aruco_parameters_comprehensive(test_image)
        print(f"\nTop 10 Results:")
        print("-" * 80)
        for i, result in enumerate(results[:10]):
            params = result['parameters']
            print(f"{i+1:2d}. Detected: {result['detected']:2d}, Rejected: {result['rejected']:2d}")
            print(f"    minPerim: {params['minMarkerPerimeterRate']:.3f}, "
                  f"maxPerim: {params['maxMarkerPerimeterRate']:.1f}")
            print(f"    winSize: {params['adaptiveThreshWinSizeMin']}-{params['adaptiveThreshWinSizeMax']}, "
                  f"polyAcc: {params['polygonalApproxAccuracyRate']:.3f}")
            print()

if __name__ == "__main__":
    main() 