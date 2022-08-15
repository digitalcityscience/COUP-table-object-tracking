from typing import Dict, List

from marker import (
    Marker,
    add_detected_markers_to_dict,
    map_detected_markers,
)
from camera import Frame, poll_frame_data
from detection import detect_markers
from image import buffer_to_array, sharpen_and_rotate_image


def track(frame: Frame, markerDict: Dict[int, Marker]):
    camera_id, image = frame
    ir_image = sharpen_and_rotate_image(buffer_to_array(image))
    corners, ids, rejectedImgPoints = detect_markers(ir_image)
    add_detected_markers_to_dict(ids, camera_id, corners, 1, markerDict)



def track_v2(frame: Frame) -> List[Marker]:
    camera_id, image = frame
    ir_image = sharpen_and_rotate_image(buffer_to_array(image))
    corners, ids, rejectedImgPoints = detect_markers(ir_image)
    marker_dict = map_detected_markers(camera_id, ids, corners)
    return list(marker_dict.values())

if __name__ == "__main__":
    for frame in poll_frame_data():
        markerDict: Dict[int, Marker] = {}
        track(frame, markerDict)
        print(markerDict)
    exit()
