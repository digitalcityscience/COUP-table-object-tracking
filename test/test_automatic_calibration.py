import numpy as np

import automatic_calibration


def _corners(center_x, center_y):
    return np.array(
        [
            [
                [center_x - 1, center_y - 1],
                [center_x + 1, center_y - 1],
                [center_x + 1, center_y + 1],
                [center_x - 1, center_y + 1],
            ]
        ],
        dtype=np.float32,
    )


def test_calibration_advances_after_one_detection_per_fixed_marker(
    monkeypatch, tmp_path
):
    image = np.zeros((10, 10), dtype=np.uint8)
    frames = iter((camera_id, image) for camera_id in ["863"] * 4 + ["104"] * 4)
    marker_ids = iter([180, 181, 182, 183, 190, 191, 192, 193])
    monkeypatch.setattr(automatic_calibration, "poll_frame_data", lambda: frames)

    def detect(_image):
        marker_id = next(marker_ids)
        return [_corners(marker_id, marker_id * 2)], np.array([marker_id]), []

    monkeypatch.setattr(automatic_calibration, "detect_markers", detect)
    monkeypatch.setattr(automatic_calibration.cv2, "imshow", lambda *args: None)
    monkeypatch.setattr(automatic_calibration.cv2, "namedWindow", lambda *args: None)
    monkeypatch.setattr(automatic_calibration.cv2, "resizeWindow", lambda *args: None)
    monkeypatch.setattr(automatic_calibration.cv2, "waitKey", lambda *_: -1)
    monkeypatch.setattr(automatic_calibration.cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(
        automatic_calibration,
        "export_pictures_for_debugging",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        automatic_calibration, "analyze_camera_distortion", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(
        automatic_calibration.cv2.aruco,
        "drawDetectedMarkers",
        lambda image, _corners, _ids: image,
    )

    result = automatic_calibration.calibrate_fixed_rig(tmp_path / "calibration.json")

    assert result["863"]["calibration_markers"]["top_left"]["pixel_position"] == [
        180.0,
        360.0,
    ]
    assert result["104"]["calibration_markers"]["bottom_left"]["pixel_position"] == [
        193.0,
        386.0,
    ]
