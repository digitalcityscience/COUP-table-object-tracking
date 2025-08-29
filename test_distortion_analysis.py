#!/usr/bin/env python3
"""
Test script for camera distortion analysis

This script demonstrates the distortion analysis functionality by:
1. Creating sample calibration data with different distortion scenarios
2. Running distortion analysis on the sample data
3. Showing how to use the distortion analysis functions

Run with: python test_distortion_analysis.py
"""

import json
import os
import numpy as np
from distortion_analysis import analyze_camera_distortion, load_and_analyze_calibration

def create_sample_calibration_data():
    """Create sample calibration data with different distortion scenarios"""
    
    # Camera 000: Perfect square (excellent calibration)
    perfect_camera = {
        "calibration_markers": {
            "top_left": {"id": "48", "pixel_position": [200, 150], "physical_position": [3, 3]},
            "top_right": {"id": "44", "pixel_position": [400, 150], "physical_position": [77, 3]},
            "bottom_right": {"id": "41", "pixel_position": [400, 350], "physical_position": [77, 77]},
            "bottom_left": {"id": "43", "pixel_position": [200, 350], "physical_position": [3, 77]}
        }
    }
    
    # Camera 001: Slightly off-center (good calibration)
    off_center_camera = {
        "calibration_markers": {
            "top_left": {"id": "68", "pixel_position": [210, 160], "physical_position": [83, 3]},
            "top_right": {"id": "62", "pixel_position": [410, 155], "physical_position": [157, 3]},
            "bottom_right": {"id": "69", "pixel_position": [405, 355], "physical_position": [157, 77]},
            "bottom_left": {"id": "60", "pixel_position": [205, 360], "physical_position": [83, 77]}
        }
    }
    
    # Camera 002: Tilted/rotated (fair calibration)
    tilted_camera = {
        "calibration_markers": {
            "top_left": {"id": "70", "pixel_position": [180, 170], "physical_position": [163, 3]},
            "top_right": {"id": "71", "pixel_position": [390, 140], "physical_position": [237, 3]},
            "bottom_right": {"id": "72", "pixel_position": [420, 340], "physical_position": [237, 77]},
            "bottom_left": {"id": "73", "pixel_position": [210, 370], "physical_position": [163, 77]}
        }
    }
    
    # Camera 003: Heavily distorted (poor calibration)
    distorted_camera = {
        "calibration_markers": {
            "top_left": {"id": "80", "pixel_position": [150, 120], "physical_position": [243, 3]},
            "top_right": {"id": "81", "pixel_position": [450, 180], "physical_position": [317, 3]},
            "bottom_right": {"id": "82", "pixel_position": [380, 380], "physical_position": [317, 77]},
            "bottom_left": {"id": "83", "pixel_position": [220, 320], "physical_position": [243, 77]}
        }
    }
    
    # Camera 004: Very poor calibration (extreme distortion)
    very_poor_camera = {
        "calibration_markers": {
            "top_left": {"id": "90", "pixel_position": [100, 100], "physical_position": [323, 3]},
            "top_right": {"id": "91", "pixel_position": [500, 200], "physical_position": [397, 3]},
            "bottom_right": {"id": "92", "pixel_position": [350, 400], "physical_position": [397, 77]},
            "bottom_left": {"id": "93", "pixel_position": [250, 250], "physical_position": [323, 77]}
        }
    }
    
    return {
        "000": perfect_camera,
        "001": off_center_camera,
        "002": tilted_camera,
        "003": distorted_camera,
        "004": very_poor_camera
    }

def save_sample_calibration(sample_data, filename="test_calibration_markers.json"):
    """Save sample calibration data to a JSON file"""
    with open(filename, 'w') as f:
        json.dump(sample_data, f, indent=2)
    print(f"Saved sample calibration data to {filename}")

def run_distortion_test():
    """Run the distortion analysis test"""
    print("="*60)
    print("CAMERA DISTORTION ANALYSIS TEST")
    print("="*60)
    
    # Create sample data
    print("\n1. Creating sample calibration data with different distortion scenarios...")
    sample_data = create_sample_calibration_data()
    
    # Save sample data
    test_file = "test_calibration_markers.json"
    save_sample_calibration(sample_data, test_file)
    
    # Run analysis on sample data
    print("\n2. Running distortion analysis on sample data...")
    try:
        results = analyze_camera_distortion(sample_data, "test_calibration_visualizations")
        
        print("\n3. Analysis Results Summary:")
        print("-" * 40)
        
        # Sort cameras by score for better presentation
        sorted_cameras = sorted(results.items(), key=lambda x: x[1]['overall_score'], reverse=True)
        
        for camera_id, result in sorted_cameras:
            print(f"\nCamera {camera_id}:")
            print(f"  Overall Score: {result['overall_score']:.3f} ({result['distortion_level']})")
            print(f"  Center Uniformity: {result['center_uniformity_score']:.3f}")
            print(f"  Side Uniformity: {result['side_uniformity_score']:.3f}")
            print(f"  Diagonal Ratio: {result['diagonal_ratio']:.3f}")
            print(f"  Angle Uniformity: {result['angle_uniformity_score']:.3f}")
            
            # Show distances from center
            distances = result['distances_from_center']
            print(f"  Distances from center: TL={distances['top_left']:.1f}, TR={distances['top_right']:.1f}, BR={distances['bottom_right']:.1f}, BL={distances['bottom_left']:.1f}")
            
            # Show key recommendations
            recommendations = result['recommendations']
            if recommendations:
                print(f"  Top recommendation: {recommendations[0]}")
        
        print(f"\n4. Visualization files saved to: test_calibration_visualizations/")
        print("   - Individual camera analysis: camera_XXX_distortion_analysis.png")
        print("   - Summary comparison: distortion_summary.png")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
    
    # Clean up
    print(f"\n5. Cleaning up test files...")
    try:
        os.remove(test_file)
        print(f"   Removed {test_file}")
    except:
        pass

def test_existing_calibration():
    """Test analysis on existing calibration file if available"""
    print("\n" + "="*60)
    print("TESTING EXISTING CALIBRATION (if available)")
    print("="*60)
    
    calibration_file = "calibration_markers.json"
    if os.path.exists(calibration_file):
        print(f"Found existing calibration file: {calibration_file}")
        try:
            results = load_and_analyze_calibration(calibration_file, "existing_calibration_analysis")
            print("✓ Analysis complete! Check existing_calibration_analysis/ for results.")
            
            # Quick summary
            if results:
                print("\nQuick Summary:")
                for camera_id, result in results.items():
                    print(f"  Camera {camera_id}: {result['overall_score']:.3f} ({result['distortion_level']})")
            
        except Exception as e:
            print(f"Error analyzing existing calibration: {e}")
    else:
        print(f"No existing calibration file found at {calibration_file}")
        print("Run find_calibration_markers.py first to create calibration data.")

def demonstrate_metrics():
    """Demonstrate what each metric measures"""
    print("\n" + "="*60)
    print("DISTORTION METRICS EXPLANATION")
    print("="*60)
    
    print("""
📏 CENTER UNIFORMITY (30% weight)
   - Measures if all markers are equidistant from the image center
   - 1.0 = Perfect: All markers same distance from center
   - <0.9 = Poor: Camera not centered over markers
   
📐 SIDE UNIFORMITY (30% weight)  
   - Measures if all sides of the quadrilateral are equal
   - 1.0 = Perfect square with equal side lengths
   - <0.9 = Poor: Camera height/angle issues
   
📏 DIAGONAL RATIO (20% weight)
   - Measures if both diagonals are equal length
   - 1.0 = Perfect: Both diagonals exactly equal
   - <0.9 = Poor: Perspective distortion
   
📐 ANGLE UNIFORMITY (20% weight)
   - Measures if all angles are 90 degrees
   - 1.0 = Perfect: All angles exactly 90°
   - <0.9 = Poor: Camera tilt/rotation issues

🎯 OVERALL SCORE
   - Weighted average of all metrics
   - 0.95+ = Excellent (green)
   - 0.90+ = Good (light green)  
   - 0.80+ = Fair (yellow)
   - 0.70+ = Poor (orange)
   - <0.70 = Very Poor (red)
   
💡 COMMON ISSUES:
   - Low center uniformity → Adjust camera position (move camera)
   - Low side uniformity → Adjust camera height/tilt
   - Low diagonal ratio → Check for lens distortion
   - Low angle uniformity → Reduce camera rotation/tilt
   - Table moved → All metrics will be poor
    """)

if __name__ == "__main__":
    try:
        # Run the main test
        # run_distortion_test()
        
        # Test existing calibration if available
        test_existing_calibration()
        
        # Show metrics explanation
        demonstrate_metrics()
        
        print("\n" + "="*60)
        print("✓ TEST COMPLETE!")
        print("="*60)
        print("\nNext steps:")
        print("1. Check the visualization files in test_calibration_visualizations/")
        print("2. Run 'python find_calibration_markers.py' to calibrate real cameras")
        print("3. The distortion analysis will run automatically after calibration")
        print("4. Use the results to adjust camera positions for optimal tracking")
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc() 