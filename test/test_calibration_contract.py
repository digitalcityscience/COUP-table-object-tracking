"""The map-calibration marker contract, and the mirror guard that protects detection."""

import json
from pathlib import Path

import pytest

from calibration_contract import (
    CORNER_ORDER,
    EXTRA_MAP_CALIBRATION_MARKER_IDS,
    MAP_CALIBRATION_MARKER_CORNERS,
    MAP_CALIBRATION_MARKER_IDS,
    assert_not_mirrored,
    corner_for_map_calibration_marker,
    is_mirrored,
    signed_area,
)

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "frontend_contract.json").read_text(
        encoding="utf-8"
    )
)


def test_calibration_marker_ids_are_the_corners_plus_the_grid():
    """200-203 are the corners; 206/207/209-213 are TOSCA-2's extra grid points (workflow step 5).

    All eleven must be here, not just the corners: `marker.py::reduceToCalibrationMarkers` waves
    exactly these ids past the confidence gate, so an id missing from this set is held back and
    never reaches the frontend to be read -- the marker would simply never appear, with nothing
    anywhere saying why. 204/205/208 were retired 2026-09-07 (they sat on the camera-stitch seam
    of a two-desk table) and must never reappear here. 214 (208's midRight seam-clearance half)
    was retired hours after shipping: that row sits at the AOI's vertical centre too, which on the
    live rig is where the ceiling beamer's projection is brightest, so a marker there is
    overexposed and undecodable regardless of seam clearance. 213, its midLeft half, reads fine
    and stays -- the centre row carries one marker, not a pair.
    """
    assert sorted(MAP_CALIBRATION_MARKER_IDS) == [200, 201, 202, 203, 206, 207, 209, 210, 211, 212]
    assert sorted(MAP_CALIBRATION_MARKER_CORNERS) == [200, 201, 202, 203]
    assert sorted(EXTRA_MAP_CALIBRATION_MARKER_IDS) == [206, 207, 209, 210, 211, 212]


def test_only_the_corner_markers_claim_a_corner():
    for marker_id in EXTRA_MAP_CALIBRATION_MARKER_IDS:
        with pytest.raises(KeyError):
            corner_for_map_calibration_marker(marker_id)


def test_corner_mapping_matches_the_frontend_contract_fixture():
    """200/201/202/203 must mean the same corners here as in TOSCA-2 and the vanilla app."""
    expected = {
        int(marker_id): corner
        for marker_id, corner in FIXTURE["map_calibration_marker_corners"].items()
        if not marker_id.startswith("_")  # `_note` is prose for a human, not a mapping
    }
    assert MAP_CALIBRATION_MARKER_CORNERS == expected
    assert expected == {
        200: "top_left",
        201: "top_right",
        202: "bottom_left",
        203: "bottom_right",
    }


def test_corner_lookup_is_by_id_not_by_iteration_order():
    for marker_id, corner in MAP_CALIBRATION_MARKER_CORNERS.items():
        assert corner_for_map_calibration_marker(marker_id) == corner
    # tolerate the string keys JSON hands back
    assert corner_for_map_calibration_marker("203") == "bottom_right"


def test_unknown_marker_id_is_rejected_loudly():
    with pytest.raises(KeyError):
        corner_for_map_calibration_marker(100)


def test_corner_order_is_a_closed_quad_traversal():
    assert CORNER_ORDER == ("top_left", "top_right", "bottom_right", "bottom_left")


def test_a_correctly_labelled_quad_is_not_mirrored():
    # y grows downward, so TL->TR->BR->BL is clockwise on screen: positive area.
    quad = [(100, 100), (900, 100), (900, 700), (100, 700)]
    assert signed_area(quad) > 0
    assert is_mirrored(quad) is False
    assert_not_mirrored(quad, context="test")


def test_a_vertically_flipped_quad_is_detected_as_mirrored():
    quad = [(100, 700), (900, 700), (900, 100), (100, 100)]
    assert is_mirrored(quad) is True
    with pytest.raises(ValueError, match="mirrored order"):
        assert_not_mirrored(quad, context="test")


def test_the_exact_quad_the_broken_calibration_produced_is_rejected():
    """Regression: this is camera 863 as `feat/automatic-camera-calibration` wrote it.

    Those four pixel positions, read in CORNER_ORDER, wind anticlockwise. The perspective
    transform built from them mirrors the stitched table image, and ArUco cannot decode a
    mirrored code -- which is why markers 200-203 were never resolved once that calibration
    file was in place.
    """
    broken_863 = [
        (311.9364929199219, 778.29736328125),    # labelled top_left
        (1041.1920166015625, 766.8277587890625),  # labelled top_right
        (1025.4012451171875, 39.21489334106445),  # labelled bottom_right
        (300.84808349609375, 54.4387321472168),   # labelled bottom_left
    ]
    assert is_mirrored(broken_863) is True
    with pytest.raises(ValueError, match="mirrored order"):
        assert_not_mirrored(broken_863, context="camera 863")


def test_the_shipped_calibration_is_the_sequential_rig_layout():
    """The target state: one sequential id block per table, corners fixed from geometry.

    Camera 863 owns 180-183 and camera 104 owns 190-193 (the arrangement the automatic
    calibration was introduced for). The corner order is clockwise from the bottom-left,
    which is how the markers are physically laid -- not the top-left the old code assumed.
    """
    path = Path(__file__).resolve().parent.parent / "calibration_markers.json"
    calibration = json.loads(path.read_text(encoding="utf-8"))
    corners_by_camera = {
        camera_id: [camera["calibration_markers"][c]["id"] for c in CORNER_ORDER]
        for camera_id, camera in calibration.items()
    }
    assert corners_by_camera == {
        "863": ["183", "182", "181", "180"],
        "104": ["193", "192", "191", "190"],
    }


def test_the_shipped_calibration_file_is_not_mirrored():
    """The calibration this server actually loads must build an unmirrored transform."""
    path = Path(__file__).resolve().parent.parent / "calibration_markers.json"
    calibration = json.loads(path.read_text(encoding="utf-8"))
    assert calibration, "calibration_markers.json is empty"
    for camera_id, camera in calibration.items():
        quad = [
            camera["calibration_markers"][corner]["pixel_position"]
            for corner in CORNER_ORDER
        ]
        assert not is_mirrored(quad), f"camera {camera_id} would mirror the table image"
