from detection import detect_markers
import os
import numpy
from image import read_from_file

fixture_file = os.path.join(
        os.path.dirname(__file__),
        "fixtures",
        "output_623_cid_130.png",
    )

def test_detect_aruco_without_matches():
  expected_rejectedImgPoints = (numpy.array([[
        [571., 478.],
        [590., 488.],
        [580., 506.],
        [563., 496.]
        ]], dtype="float32"), 
  numpy.array([[[752., 161.],
        [760., 177.],
        [743., 186.],
        [735., 169.]]], dtype="float32"))
        
  corners, ids, rejectedImgPoints    = detect_markers(read_from_file(fixture_file))

  assert corners == ()
  assert numpy.array_equal(rejectedImgPoints, expected_rejectedImgPoints)
  assert ids == None