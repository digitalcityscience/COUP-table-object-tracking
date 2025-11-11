import json
import os
from typing import Dict

from camera_setup import prompt_camera_setup
from save_calibration_markers import save_calibration_markers


def check_calibration_exists(file_path: str = "calibration_markers.json") -> bool:
    """
    Check if calibration file exists and contains valid data
    
    Args:
        file_path: Path to the calibration file
        
    Returns:
        bool: True if calibration exists and is valid, False otherwise
    """
    if not os.path.exists(file_path):
        print(f"❌ No calibration file found at {file_path}")
        return False
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        # Basic validation - check if we have camera data
        if not data or len(data) == 0:
            print(f"❌ Calibration file {file_path} is empty")
            return False
            
        # Check if each camera has required calibration markers
        for camera_id, camera_data in data.items():
            if "calibration_markers" not in camera_data:
                print(f"❌ Camera {camera_id} missing calibration markers")
                return False
                
            markers = camera_data["calibration_markers"]
            required_positions = ["top_left", "top_right", "bottom_right", "bottom_left"]
            
            for pos in required_positions:
                if pos not in markers:
                    print(f"❌ Camera {camera_id} missing {pos} marker")
                    return False
                if "pixel_position" not in markers[pos]:
                    print(f"❌ Camera {camera_id} {pos} marker missing pixel position")
                    return False
                    
        print(f"✅ Valid calibration found with {len(data)} cameras")
        return True
        
    except json.JSONDecodeError:
        print(f"❌ Calibration file {file_path} is corrupted (invalid JSON)")
        return False
    except Exception as e:
        print(f"❌ Error reading calibration file {file_path}: {e}")
        return False



def run_initial_calibration_if_needed():
    # Step 1: Check if calibration already exists
    if check_calibration_exists():
        print("Calibration file found, skipping initial calibration")
        return
    
    """
    Run the initial calibration setup with enhanced visual guidance if needed
    
    This function:
    1. Checks if calibration already exists
    2. Prompts user for recalibration decision if it exists
    3. Runs enhanced camera setup with visual orientation guides
    4. Performs calibration marker detection and saves results
    """

    print("\n📝 NO EXISTING CALIBRATION FOUND")
    print("Running initial calibration setup with enhanced visual guides...")
    
    
    print("\n" + "="*80)
    print("🚀 CALIBRATION SYSTEM - ENHANCED WITH VISUAL GUIDES")
    
        
    # Step 2: Run enhanced camera setup
    print("\n" + "="*80)
    print("🎯 STEP 1: ENHANCED CAMERA SETUP")
    print("="*80)
    print("Setting up cameras with visual orientation guides...")
    
    camera_setup = prompt_camera_setup()
    
    # Step 3: Perform calibration marker detection
    print("\n" + "="*80)
    print("🎯 STEP 2: CALIBRATION MARKER DETECTION")
    print("="*80)
    print("Now detecting and saving calibration markers...")
    print("This process will use the coordinate system established in Step 1.")
    
    save_calibration_markers(camera_setup)
    
    print("\n" + "="*80)
    print("🎉 CALIBRATION COMPLETE!")
    print("="*80)
    print("✅ Camera setup completed with enhanced visual guidance")
    print("✅ Calibration markers detected and saved")
    print("✅ Coordinate system properly established")
    print("✅ Ready for real-time object tracking")
    print()
    print("📁 Calibration data saved to: calibration_markers.json")
    print("📁 Visual reports saved to: calibration_visualizations/")
    print("="*80)


def load_calibration_markers(file_path: str) -> Dict:
    """
    Load calibration markers from JSON file
    
    Args:
        file_path: Path to the calibration markers JSON file
        
    Returns:
        Dict: Calibration data containing camera configurations and marker positions
        
    Raises:
        FileNotFoundError: If calibration file doesn't exist
        json.JSONDecodeError: If calibration file is corrupted
    """
    try:
        with open(file_path, 'r') as f:
            calibration_data = json.load(f)
        
        print(f"📄 Loaded calibration data for {len(calibration_data)} cameras from {file_path}")
        return calibration_data
        
    except FileNotFoundError:
        print(f"❌ ERROR: Calibration file not found: {file_path}")
        print("   Please run calibration first by restarting the server.")
        raise
    except json.JSONDecodeError as e:
        print(f"❌ ERROR: Corrupted calibration file: {file_path}")
        print(f"   JSON decode error: {e}")
        print("   Please delete the file and run calibration again.")
        raise
    except Exception as e:
        print(f"❌ ERROR: Failed to load calibration file: {file_path}")
        print(f"   Error: {e}")
        raise
