import math
import os

import numpy
import pytest
from detection import detect_markers, normalizeCorners
from image import read_from_file

fixture_file = os.path.join(
    os.path.dirname(__file__),
    "fixtures",
    "output_623_cid_130.png",
)


def test_detect_aruco_without_matches():
    expected_rejectedImgPoints = (
        numpy.array(
            [[[571.0, 478.0], [590.0, 488.0], [580.0, 506.0], [563.0, 496.0]]],
            dtype="float32",
        ),
        numpy.array(
            [[[752.0, 161.0], [760.0, 177.0], [743.0, 186.0], [735.0, 169.0]]],
            dtype="float32",
        ),
    )

    corners, ids, rejectedImgPoints = detect_markers(read_from_file(fixture_file))
    assert corners == ()
    assert numpy.array_equal(rejectedImgPoints, expected_rejectedImgPoints)
    assert ids == None


fixture_file_sharp = os.path.join(
    os.path.dirname(__file__),
    "fixtures",
    "sharpened_output_623_cid_130.png",
)


def test_detect_aruco_with_matches():
    expected_marker_ids = numpy.array([[19], [20]])
    _, ids, _ = detect_markers(read_from_file(fixture_file_sharp))
    assert numpy.array_equiv(ids, expected_marker_ids)


def test_normalize_corners():
    corners = numpy.array(
        [
            [
                    [453.17453, 631.852],
                    [439.2763, 619.1162],
                    [446.3173, 596.88025],
                    [460.5247, 608.08624],
            ]
        ],
        dtype="float32",
    )
    assert numpy.array_equiv(normalizeCorners(corners), [449, 614, 78.90624111411])


def _square_at(center_x, center_y, rotation_deg, half=40.0):
    """One marker quad, rotated `rotation_deg` counter-clockwise in the table frame."""
    angle = math.radians(rotation_deg)
    cos, sin = math.cos(angle), math.sin(angle)
    return numpy.array(
        [
            [
                [
                    center_x + half * (dx * cos - dy * sin),
                    center_y + half * (dx * sin + dy * cos),
                ]
                for dx, dy in ((1, 1), (-1, 1), (-1, -1), (1, -1))
            ]
        ],
        dtype="float32",
    )


def test_normalize_corners_reports_rotation_counter_clockwise_positive():
    """The table frame is +x East, +y North -- turning a block left must read as a bigger angle.

    This is the regression guard for the sign bug fixed on 2026-09-04: a stray `-1` in
    `normalizeCorners` made the reported angle run backwards, so a block turned +90 deg on the
    table placed its footprint at -90 deg on the map. `place_geometry` consumes this number with
    a counter-clockwise-positive rotation matrix, so the producer has to agree.
    """
    baseline = normalizeCorners(_square_at(500.0, 500.0, 0.0))[2]

    for turn in (30.0, 90.0, 170.0):
        turned = normalizeCorners(_square_at(500.0, 500.0, turn))[2]
        assert (turned - baseline + 180.0) % 360.0 - 180.0 == pytest.approx(turn, abs=1e-3)
