import os

import numpy
from image import read_from_file, sharpen_and_rotate_image

fixture_file = os.path.join(
    os.path.dirname(__file__),
    "fixtures",
    "output_623_cid_130.png",
)


def test_sharpen_and_rotate_image():
    fixture = read_from_file(fixture_file)
    given = sharpen_and_rotate_image(fixture)
    expected = read_from_file(
        os.path.join(
            os.path.dirname(__file__),
            "fixtures",
            "sharpened_output_623_cid_130.png",
        )
    )
    assert numpy.array_equal(given, expected)
