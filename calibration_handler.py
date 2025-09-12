import json
import time
import os
from typing import Dict

from camera_setup import prompt_camera_setup
from save_calibration_markers import save_calibration_markers

def check_calibration_exists(file_path: str = "calibration_markers.json") -> bool:
    """
    Check if calibration file exists with valid pixel positions
    
    Returns:
        True if calibration exists and is complete, False otherwise
    """
    if not os.path.exists(file_path):
        print(f"No existing calibration file found at {file_path}")
        return False
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Check if all cameras have pixel positions
        for camera_id, camera_data in data.items():
            if "calibration_markers" not in camera_data:
                return False
            
            for position, marker_info in camera_data["calibration_markers"].items():
                if marker_info.get("pixel_position") is None:
                    print(f"Camera {camera_id} marker at {position} missing pixel position")
                    return False
        
        print(f"Found complete calibration for {len(data)} cameras")
        return True
        
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error reading calibration file: {e}")
        return False
    

def prompt_recalibration():    
    user_input = input("Enter 1 to recalibrate")
    if user_input == '1':
        print("Recalibrating...")
        return True
    else:
        return False




def run_initial_calibration_if_needed():
    # Step 1: Run calibration setup
    print("Step 1: Running calibration setup...")
    if check_calibration_exists():
        print("Existing calibration found.")
        if not prompt_recalibration():
            return
        
        print("OVERRIDING EXISITING CALIBRATION")
            
    camera_setup = prompt_camera_setup()        
    save_calibration_markers(camera_setup)


    
def load_calibration_markers(file_path: str) -> Dict:
    """Load calibration markers from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)
