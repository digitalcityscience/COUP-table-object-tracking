from camera import get_device_manager
from show_camera_streams import show_camera_streams


def choose_camera_position(cam_id):
    """
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


def prompt_calibration_marker_ids():
    markers = {}
    for pos in ["top_left", "top_right", "bottom_right", "bottom_left"]:
       marker_id = input(f"  {pos} marker ID: ").strip()
       if not marker_id:
           raise ValueError(f"Marker ID for {pos} cannot be empty")
       markers[pos] = {"id": marker_id}

    return markers



def add_physical_position_of_calibration_marker(markers, table_width, table_height, marker_offset):
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

    device_manager = get_device_manager()
    camera_ids = [id[-3:] for id in device_manager.get_enabled_devices_ids()]

    # Get table size
    table_width = float(input("Table element width (cm): "))
    table_height = float(input("Table element height (cm): "))
    
    # Get marker offset
    marker_offset = float(input("Distance from marker to table edge (cm): "))

    table_element_measurements = {"width": table_width, "height": table_height, "marker_offset": marker_offset}

    camera_setup = {cam_id: {"position": None, "calibration_markers": {}, "measurements": table_element_measurements} for cam_id in camera_ids}

    print("PLEASE NOTE CAMERA ID OF LEFT CAMERA")
    print("PLEASE NOTE CAMERA ID OF RIGHT CAMERA")
    print("PRESS Q TO QUIT")

    show_camera_streams()

    # Ask the user to enter the camera positions
    for cam_id in camera_ids:
        camera_setup[cam_id]["position"] = choose_camera_position(cam_id)
    
    print(f"Camera positions are: {camera_setup}")
    

    for cam_id in camera_ids:
        # ask the user to enter the calibration marker ids
        print(f"Enter calibration marker ids for camera {cam_id}")
        calibration_markers = prompt_calibration_marker_ids()
        calibration_markers = add_physical_position_of_calibration_marker(
            calibration_markers,
            table_width=table_width,
            table_height=table_height,
            marker_offset=marker_offset
        )

        camera_setup[cam_id]["calibration_markers"] = calibration_markers

   
    print(f"camera setup: {camera_setup} " )
    return camera_setup