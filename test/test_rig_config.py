import json
from pathlib import Path

import pytest

from calibration_contract import CORNER_ORDER, is_mirrored
from rig_config import (
    RIG_CAMERAS,
    assign_corners_by_geometry,
    build_fixed_camera_setup,
)


def test_fixed_camera_setup_describes_the_table_but_not_the_corner_layout():
    """Physical geometry is known up front; which id is at which corner is measured.

    The rig used to declare a corner->id table here. It was a guess (each camera getting a
    contiguous, clockwise-ordered block) and it did not match the real rig, which is how the
    corner labels ended up vertically flipped.
    """
    setup = build_fixed_camera_setup()

    assert setup["863"]["position"] == "top_left"
    assert setup["104"]["position"] == "top_right"
    assert setup["863"]["measurements"] == {
        "width": 80.0,
        "height": 80.0,
        "marker_offset": 3.0,
    }
    assert setup["104"]["calibration_markers"]["bottom_right"]["physical_position"] == [
        77.0,
        77.0,
    ]
    for camera in setup.values():
        for corner in CORNER_ORDER:
            marker = camera["calibration_markers"][corner]
            assert "physical_position" in marker
            assert "id" not in marker, "corner ids must be measured, never assumed"
            assert "pixel_position" not in marker


def test_rig_declares_each_camera_s_marker_ids_as_an_unordered_set():
    assert set(RIG_CAMERAS["863"]["marker_ids"]) == {180, 182, 192, 193}
    assert set(RIG_CAMERAS["104"]["marker_ids"]) == {181, 183, 190, 191}


def test_rig_marker_ids_match_the_last_known_good_calibration():
    """The declared per-camera id sets come from the measured calibration, not invention."""
    path = Path(__file__).resolve().parent.parent / "calibration_markers.json"
    calibration = json.loads(path.read_text(encoding="utf-8"))
    for camera_id, camera in calibration.items():
        measured = {
            int(camera["calibration_markers"][corner]["id"]) for corner in CORNER_ORDER
        }
        assert measured == set(RIG_CAMERAS[camera_id]["marker_ids"])


def test_corners_are_assigned_from_observed_positions():
    observed = {
        193: (1019.7, 31.6),   # top-right of the frame
        180: (299.9, 43.9),    # top-left
        182: (313.9, 769.7),   # bottom-left
        192: (1039.9, 758.1),  # bottom-right
    }
    assert assign_corners_by_geometry(observed) == {
        "top_left": 180,
        "top_right": 193,
        "bottom_right": 192,
        "bottom_left": 182,
    }


def test_corner_assignment_is_independent_of_detection_order():
    """Detection order, and therefore dict insertion order, must not change the result."""
    positions = {
        180: (299.9, 43.9),
        193: (1019.7, 31.6),
        192: (1039.9, 758.1),
        182: (313.9, 769.7),
    }
    expected = assign_corners_by_geometry(dict(positions))
    for rotation in range(len(positions)):
        items = list(positions.items())
        shuffled = dict(items[rotation:] + items[:rotation])
        assert assign_corners_by_geometry(shuffled) == expected
    assert assign_corners_by_geometry(dict(reversed(list(positions.items())))) == expected


def test_corner_assignment_never_produces_a_mirrored_quad():
    observed = {
        180: (299.9, 43.9),
        193: (1019.7, 31.6),
        192: (1039.9, 758.1),
        182: (313.9, 769.7),
    }
    corners = assign_corners_by_geometry(observed)
    quad = [observed[corners[corner]] for corner in CORNER_ORDER]
    assert not is_mirrored(quad)


def test_corner_assignment_requires_all_four_markers():
    with pytest.raises(ValueError, match="exactly 4"):
        assign_corners_by_geometry({180: (10, 10), 181: (20, 10)})
