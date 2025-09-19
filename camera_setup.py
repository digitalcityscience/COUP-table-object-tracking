from camera import get_device_manager
from show_camera_streams import show_single_camera_with_guide, show_camera_streams


def choose_camera_position_with_visual_guide(cam_id):
    """
    Enhanced camera position selection with clear instructions and visual guidance
    """
    print(f"\n" + "="*70)
    print(f"🎯 CAMERA {cam_id} POSITION SETUP")
    print("="*70)
    
    # Show the specific camera with orientation guide
    show_single_camera_with_guide(cam_id, duration_seconds=8)
    
    print("\n📋 INSTRUCTIONS:")
    print("You just saw the camera view with orientation arrows and corner labels.")
    print("Now select where THIS CAMERA is physically positioned relative to your table:")
    print()
    print("Physical camera positions:")
    print("  1. top_left     → Camera sees the TOP-LEFT portion of the table")
    print("  2. top_right    → Camera sees the TOP-RIGHT portion of the table") 
    print("  3. bottom_left  → Camera sees the BOTTOM-LEFT portion of the table")
    print("  4. bottom_right → Camera sees the BOTTOM-RIGHT portion of the table")
    print()
    print("💡 TIP: Match the camera's physical location to what it can see!")
    
    options = ["top_left", "top_right", "bottom_left", "bottom_right"]
    
    while True:
        try:
            choice = int(input(f"\nEnter position for camera {cam_id} (1-4): "))
            if 1 <= choice <= len(options):
                selected = options[choice - 1]
                print(f"✅ Camera {cam_id} assigned to {selected}")
                return selected
            else:
                print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")
        except ValueError:
            print("❌ Please enter a valid number (1-4).")


def choose_camera_position(cam_id):
    """
    Legacy function - kept for backwards compatibility
    Asks the user to identify the physical camera position. Is it the left or right side of the table?
    If we have only 2 table cubes, select top_left and top_right
    """
    options = ["top_left", "top_right", "bottom_left", "bottom_right"]
    
    print(f"Where is camera {cam_id} located?")
    print("Please choose one of the following options:")
    for index, option in enumerate(options, start=1):
        print(f"{index}. {option}")

    while True:
        try:
            choice = int(input("Enter the number of your choice: "))
            if 1 <= choice <= len(options):
                return options[choice - 1]
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")


def prompt_calibration_marker_ids_with_validation(cam_id):
    """
    Enhanced marker ID input with visual validation and clear instructions
    """
    print(f"\n" + "="*70)
    print(f"🎯 CALIBRATION MARKERS FOR CAMERA {cam_id}")
    print("="*70)
    
    # Show the camera again so user can see the markers
    print("Showing camera view again so you can identify marker IDs...")
    show_single_camera_with_guide(cam_id, duration_seconds=10)
    
    print("\n📋 MARKER ID SETUP:")
    print("Look at the camera window you just saw and identify the ArUco marker IDs")
    print("in each corner position AS YOU SEE THEM in the camera image.")
    print()
    print("The system will map these to physical coordinates automatically.")
    print("Just tell us which marker ID is in each corner of the camera view:")
    print()
    
    markers = {}
    positions = ["top_left", "top_right", "bottom_right", "bottom_left"]
    
    for pos in positions:
        while True:
            marker_id = input(f"  📍 {pos.replace('_', '-').upper()} corner marker ID: ").strip()
            if not marker_id:
                print("     ❌ Please enter a marker ID.")
                continue
            
            # Validate that it's a number
            try:
                int(marker_id)
                markers[pos] = {"id": marker_id}
                print(f"     ✅ {pos.replace('_', '-').upper()}: ID {marker_id}")
                break
            except ValueError:
                print("     ❌ Please enter a numeric marker ID.")
    
    print(f"\n✅ All markers configured for camera {cam_id}!")
    return markers


def prompt_calibration_marker_ids():
    """
    Legacy function - kept for backwards compatibility
    """
    markers = {}
    for pos in ["top_left", "top_right", "bottom_right", "bottom_left"]:
       marker_id = input(f"  {pos} marker ID: ").strip()
       if not marker_id:
           print("Please enter a marker ID.")
           return prompt_calibration_marker_ids()
       markers[pos] = {"id": marker_id}

    return markers



def add_physical_position_of_calibration_marker(markers, table_width, table_height, marker_offset):
    """
    Maps marker IDs to their physical positions on the table
    
    Physical coordinate system:
    - Origin (0,0) is at the physical top-left corner of the table
    - X increases going right
    - Y increases going down  
    - Units are in centimeters
    """
    return {
                "top_left": {
                    "id": markers["top_left"]["id"],
                    "physical_position": [marker_offset, marker_offset]
                },
                "top_right": {
                    "id": markers["top_right"]["id"],
                    "physical_position": [table_width - marker_offset, marker_offset]
                },
                "bottom_right": {
                    "id": markers["bottom_right"]["id"],
                    "physical_position": [table_width - marker_offset, table_height - marker_offset]
                },
                "bottom_left": {
                    "id": markers["bottom_left"]["id"],
                    "physical_position": [marker_offset, table_height - marker_offset]
                }
            }
        

def prompt_camera_setup():
    """
    Enhanced camera setup with visual orientation guides
    """
    print("\n" + "="*80)
    print("🚀 ENHANCED CAMERA CALIBRATION SETUP")
    print("="*80)
    print("This setup will help you correctly configure your cameras with visual guides")
    print("to prevent coordinate system mismatches and image flipping issues.")
    print("="*80)

    device_manager = get_device_manager()
    camera_ids = [id[-3:] for id in device_manager.get_enabled_devices_ids()]

    print(f"\n📷 Detected {len(camera_ids)} cameras: {camera_ids}")

    # Get table size
    print("\n📏 TABLE MEASUREMENTS:")
    table_width = float(input("Table element width (cm): "))
    table_height = float(input("Table element height (cm): "))
    
    # Get marker offset
    marker_offset = float(input("Distance from marker to table edge (cm): "))

    table_element_measurements = {"width": table_width, "height": table_height, "marker_offset": marker_offset}

    camera_setup = {cam_id: {"position": None, "calibration_markers": {}, "measurements": table_element_measurements} for cam_id in camera_ids}

    # Show all cameras first for overview
    print(f"\n📹 CAMERA OVERVIEW:")
    print("First, let's look at all cameras together to understand the setup...")
    print("Press 'Q' to close the overview and continue with individual camera setup.")
    
    show_camera_streams()

    # Configure each camera individually with enhanced guidance
    print(f"\n🔧 INDIVIDUAL CAMERA SETUP:")
    print("Now we'll configure each camera individually with visual guides...")
    
    for cam_id in camera_ids:
        print(f"\n" + "-"*50)
        print(f"Setting up Camera {cam_id} ({camera_ids.index(cam_id)+1} of {len(camera_ids)})")
        print("-"*50)
        
        # Enhanced camera position selection with visual guide
        camera_setup[cam_id]["position"] = choose_camera_position_with_visual_guide(cam_id)
    
    print(f"\n✅ Camera positions configured:")
    for cam_id, config in camera_setup.items():
        print(f"   Camera {cam_id}: {config['position']}")

    # Configure calibration markers for each camera
    print(f"\n🎯 CALIBRATION MARKER SETUP:")
    print("Now we'll identify the ArUco markers for each camera...")
    
    for cam_id in camera_ids:
        print(f"\n" + "-"*50)
        print(f"Configuring markers for Camera {cam_id} ({camera_ids.index(cam_id)+1} of {len(camera_ids)})")
        print("-"*50)
        
        # Enhanced marker ID input with validation
        calibration_markers = prompt_calibration_marker_ids_with_validation(cam_id)
        calibration_markers = add_physical_position_of_calibration_marker(
            calibration_markers,
            table_width=table_width,
            table_height=table_height,
            marker_offset=marker_offset
        )

        camera_setup[cam_id]["calibration_markers"] = calibration_markers

    print(f"\n🎉 CAMERA SETUP COMPLETE!")
    print("="*80)
    print("✅ All cameras configured with enhanced visual guidance")
    print("✅ Coordinate system properly established")
    print("✅ Ready for calibration marker detection")
    print("="*80)
    
    return camera_setup
