import numpy as np
import pytest

import automatic_calibration
from calibration_contract import CORNER_ORDER, is_mirrored


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


def _silence_ui(monkeypatch):
    for name in ("imshow", "namedWindow", "resizeWindow", "destroyAllWindows"):
        monkeypatch.setattr(automatic_calibration.cv2, name, lambda *a, **k: None)
    monkeypatch.setattr(automatic_calibration.cv2, "waitKey", lambda *_: -1)
    monkeypatch.setattr(automatic_calibration.cv2, "putText", lambda *a, **k: None)
    monkeypatch.setattr(
        automatic_calibration.cv2.aruco,
        "drawDetectedMarkers",
        lambda image, _corners, _ids: image,
    )
    monkeypatch.setattr(
        automatic_calibration, "export_pictures_for_debugging", lambda *a, **k: None
    )
    monkeypatch.setattr(
        automatic_calibration, "analyze_camera_distortion", lambda *a, **k: {}
    )


#: Where each rig marker really sits in its camera's frame, as measured on the live rig
#: (calibration_markers.json on feat/physical-building-catalog). Note the layout is
#: interleaved -- camera 863 sees 180/193/192/182 -- which is exactly what the old
#: hardcoded "contiguous clockwise block" assumption got wrong.
RIG_LAYOUT = {
    "863": {180: (299.9, 43.9), 193: (1019.7, 31.6), 192: (1039.9, 758.1), 182: (313.9, 769.7)},
    "104": {190: (335.3, 33.6), 181: (1051.3, 47.1), 183: (1063.4, 764.7), 191: (321.4, 777.5)},
}


def _run(monkeypatch, tmp_path, layout, detection_order=None, timeout_seconds=1.0):
    """Drive calibrate_fixed_rig against a synthetic rig with a known layout."""
    image = np.zeros((800, 1280), dtype=np.uint8)
    state = {"camera": None}

    def frames():
        # Round-robin forever, exactly like the real camera poll: the code under test is
        # responsible for skipping frames from the camera it is not currently calibrating.
        while True:
            for camera_id in ("863", "104"):
                state["camera"] = camera_id
                yield camera_id, image

    monkeypatch.setattr(automatic_calibration, "poll_frame_data", frames)
    _silence_ui(monkeypatch)

    def detect(_image):
        # Report every marker visible to whichever camera produced this frame.
        markers = layout[state["camera"]]
        order = (
            detection_order[state["camera"]] if detection_order else list(markers)
        )
        order = [marker_id for marker_id in order if marker_id in markers]
        if not order:
            return [], None, []
        ids = np.array([[marker_id] for marker_id in order])
        corners = [_corners(*markers[marker_id]) for marker_id in order]
        return corners, ids, []

    monkeypatch.setattr(automatic_calibration, "detect_markers", detect)
    return automatic_calibration.calibrate_fixed_rig(
        tmp_path / "calibration.json", timeout_seconds=timeout_seconds
    )


def test_corner_roles_come_from_where_the_markers_actually_are(monkeypatch, tmp_path):
    result = _run(monkeypatch, tmp_path, RIG_LAYOUT)

    ids_863 = [
        result["863"]["calibration_markers"][corner]["id"] for corner in CORNER_ORDER
    ]
    ids_104 = [
        result["104"]["calibration_markers"][corner]["id"] for corner in CORNER_ORDER
    ]
    # The real, interleaved rig layout -- not the contiguous block the old code assumed.
    assert ids_863 == ["180", "193", "192", "182"]
    assert ids_104 == ["190", "181", "183", "191"]


def test_produced_calibration_is_never_mirrored(monkeypatch, tmp_path):
    """The regression guard: a mirrored calibration silently kills ArUco detection."""
    result = _run(monkeypatch, tmp_path, RIG_LAYOUT)
    for camera_id, camera in result.items():
        quad = [
            camera["calibration_markers"][corner]["pixel_position"]
            for corner in CORNER_ORDER
        ]
        assert not is_mirrored(quad), f"camera {camera_id} mirrors the table image"


def test_result_does_not_depend_on_the_order_markers_are_detected_in(
    monkeypatch, tmp_path
):
    forward = _run(monkeypatch, tmp_path, RIG_LAYOUT)
    reversed_order = {
        camera_id: list(reversed(list(markers)))
        for camera_id, markers in RIG_LAYOUT.items()
    }
    backward = _run(monkeypatch, tmp_path, RIG_LAYOUT, detection_order=reversed_order)

    for camera_id in RIG_LAYOUT:
        assert [
            forward[camera_id]["calibration_markers"][c]["id"] for c in CORNER_ORDER
        ] == [
            backward[camera_id]["calibration_markers"][c]["id"] for c in CORNER_ORDER
        ]


def test_pixel_positions_are_the_measured_centres(monkeypatch, tmp_path):
    result = _run(monkeypatch, tmp_path, RIG_LAYOUT)
    top_left = result["863"]["calibration_markers"]["top_left"]
    assert top_left["id"] == "180"
    assert top_left["pixel_position"] == pytest.approx([299.9, 43.9], abs=1e-3)


def test_a_camera_that_never_shows_all_four_markers_fails_loudly(
    monkeypatch, tmp_path
):
    partial = {
        "863": {180: (299.9, 43.9), 193: (1019.7, 31.6)},
        "104": RIG_LAYOUT["104"],
    }
    with pytest.raises(RuntimeError, match="timed out"):
        _run(monkeypatch, tmp_path, partial)
