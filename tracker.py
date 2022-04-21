from typing import Dict

from building import (
    Building,
    add_detected_buildings_to_dict,
)
from camera import Frame, poll_frame_data
from detection import detect_markers
from image import buffer_to_array, sharpen_and_rotate_image


def track(frame: Frame, buildingDict: Dict[int, Building]):
    camera_id, image = frame
    ir_image = sharpen_and_rotate_image(buffer_to_array(image))
    corners, ids, rejectedImgPoints = detect_markers(ir_image)
    add_detected_buildings_to_dict(ids, camera_id, corners, 1, buildingDict)


if __name__ == "__main__":
    for frame in poll_frame_data():
        buildingDict: Dict[int, Building] = {}
        track(frame, buildingDict)
        print(buildingDict)
    exit()
