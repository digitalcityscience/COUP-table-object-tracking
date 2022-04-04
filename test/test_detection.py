import os

import numpy
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
                [
                    [453.17453, 631.852],
                    [439.2763, 619.1162],
                    [446.3173, 596.88025],
                    [460.5247, 608.08624],
                ]
            ]
        ],
        dtype="float32",
    )
    assert numpy.array_equiv(normalizeCorners(corners), [449, 614, 78.90624111411])
