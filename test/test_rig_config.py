import json
from pathlib import Path

import pytest

from calibration_contract import CORNER_ORDER, is_mirrored
from rig_config import (
    MARKERS_PER_CAMERA,
    MIN_QUAD_FILL_RATIO,
    RIG_CAMERAS,
    RIG_MARKER_IDS,
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


def test_each_table_owns_its_own_sequential_marker_block():
    """863 -> 180-183, 104 -> 190-193: the operator's arrangement, stated once.

    This has to stay per-camera rather than one shared pool. The cameras' views overlap,
    so camera 104 can see 863's markers -- on the live rig it read 183 in the middle of
    its own frame. With a shared pool, "whichever four turn up" swapped a missing 192 for
    that foreign 183 and calibrated against a nonsense quad.
    """
    assert RIG_CAMERAS["863"]["marker_ids"] == (180, 181, 182, 183)
    assert RIG_CAMERAS["104"]["marker_ids"] == (190, 191, 192, 193)
    assert RIG_MARKER_IDS == {180, 181, 182, 183, 190, 191, 192, 193}
    assert MARKERS_PER_CAMERA == 4
    # ...but no corner is assigned here; that is measured.
    for camera in RIG_CAMERAS.values():
        assert set(camera) == {"position", "marker_ids"}


#: Exactly what camera 863 read on the live rig.
LIVE_863 = {181: (1040.6, 766.3), 182: (1025.2, 39.9), 183: (301.4, 53.8), 180: (313.1, 777.5)}
#: Camera 104's own four, correctly read.
LIVE_104 = {193: (339.0, 35.7), 192: (1047.2, 49.7), 191: (1057.9, 766.8), 190: (317.0, 783.0)}
#: Camera 104 as it actually went wrong: 192 occluded, foreign 183 at frame centre.
LIVE_104_WITH_FOREIGN_MARKER = {
    193: (340.5, 38.1), 191: (1056.5, 766.3), 183: (604.5, 393.7), 190: (317.7, 782.5),
}


def test_live_rig_readings_resolve_to_the_right_corners():
    assert assign_corners_by_geometry(LIVE_863, context="camera 863") == {
        "top_left": 183, "top_right": 182, "bottom_right": 181, "bottom_left": 180,
    }
    assert assign_corners_by_geometry(LIVE_104, context="camera 104") == {
        "top_left": 193, "top_right": 192, "bottom_right": 191, "bottom_left": 190,
    }


def test_a_stray_marker_in_the_middle_of_the_frame_is_rejected():
    """The exact set that produced a silent, nonsense calibration on the live rig.

    Four table corners fill ~0.96 of their bounding box; this set fills 0.44. Before the
    fill check it was accepted, assigning the foreign 183 as top_right.
    """
    with pytest.raises(ValueError, match="do not outline a table"):
        assign_corners_by_geometry(LIVE_104_WITH_FOREIGN_MARKER, context="camera 104")


def test_genuine_reads_clear_the_fill_threshold_with_margin():
    for label, observed in (("863", LIVE_863), ("104", LIVE_104)):
        quad = [observed[i] for i in assign_corners_by_geometry(observed).values()]
        xs, ys = [p[0] for p in quad], [p[1] for p in quad]
        area = abs(0.5 * sum(
            quad[i][0] * quad[(i + 1) % 4][1] - quad[(i + 1) % 4][0] * quad[i][1]
            for i in range(4)
        ))
        fill = area / ((max(xs) - min(xs)) * (max(ys) - min(ys)))
        assert fill > MIN_QUAD_FILL_RATIO + 0.15, f"camera {label} fill {fill:.3f} too close to the limit"


def test_the_shipped_calibration_uses_each_camera_s_own_markers():
    path = Path(__file__).resolve().parent.parent / "calibration_markers.json"
    calibration = json.loads(path.read_text(encoding="utf-8"))
    for camera_id, camera in calibration.items():
        measured = {
            int(camera["calibration_markers"][corner]["id"]) for corner in CORNER_ORDER
        }
        assert len(measured) == MARKERS_PER_CAMERA
        assert measured == set(RIG_CAMERAS[camera_id]["marker_ids"]), (
            f"camera {camera_id} calibrated against another table's markers"
        )


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
