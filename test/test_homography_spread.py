"""What adding calibration points actually buys, measured rather than asserted.

Workflow step 5's claim is that the four corner markers -- which on the 2026-08-31 rig landed in
the middle two-thirds of the stitched image, `(288,658) (1347,663) (298,153) (1350,150)` out of
1600x800 -- leave the whole outer third of the table on extrapolation. `create_basemap_homography`
is untouched: it has accepted `>= 4` points from the start and switches to least-squares above
four all by itself. These tests pin down that the extra points are worth projecting.
"""

from pathlib import Path
import sys

import numpy as np
import pytest
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pixel_to_utm import (
    BasemapCalibrationPoint,
    create_basemap_homography,
    project_pixels_to_utm,
)

_UTM_TO_WGS84 = Transformer.from_crs("EPSG:25832", "EPSG:4326", always_xy=True)
_UTM_ORIGIN = (566_000.0, 5_936_000.0)
_METRES_PER_PIXEL = 0.30

#: The stitched table image the rig produces.
_TABLE_WIDTH_PX, _TABLE_HEIGHT_PX = 1600.0, 800.0

#: The four corner markers, where they actually landed on 2026-08-31 -- the middle two-thirds.
_MEASURED_CORNERS = [(288.0, 658.0), (1347.0, 663.0), (298.0, 153.0), (1350.0, 150.0)]

#: The same four, plus the five grid markers TOSCA-2 now projects (edge midpoints and centre).
_GRID_EXTRAS = [
    (819.0, 156.0),
    (817.0, 660.0),
    (293.0, 405.0),
    (1348.0, 406.0),
    (818.0, 408.0),
]


def _true_utm(pixel: tuple[float, float]) -> tuple[float, float]:
    """The exact UTM position a pixel maps to under a clean, purely affine table."""
    return (
        _UTM_ORIGIN[0] + pixel[0] * _METRES_PER_PIXEL,
        _UTM_ORIGIN[1] - pixel[1] * _METRES_PER_PIXEL,
    )


def _calibration_point(pixel, noise=(0.0, 0.0)) -> BasemapCalibrationPoint:
    """A correspondence whose *pixel* reading is off by `noise`, as a real detection is."""
    easting, northing = _true_utm(pixel)
    longitude, latitude = _UTM_TO_WGS84.transform(easting, northing)
    return BasemapCalibrationPoint(
        pixel_position=(pixel[0] + noise[0], pixel[1] + noise[1]),
        lat_lon_position=(latitude, longitude),
    )


def _worst_error_metres(homography, pixels) -> float:
    """How far the homography puts each pixel from where it *truly* belongs.

    Measured against ground truth, not against the correspondences, so a fit that absorbed its
    own noise is scored on the damage that noise does everywhere else -- which is the only
    question worth asking about the parts of the table nobody calibrated on.
    """
    projected = project_pixels_to_utm(homography, np.array(pixels, dtype=np.float64))
    truth = np.array([_true_utm(pixel) for pixel in pixels], dtype=np.float64)
    return float(np.max(np.hypot(*(projected - truth).T)))


def _worst_reprojection_residual_metres(homography, points) -> float:
    """How far the fit misses its *own* correspondences -- the number a calibration can self-report.

    Deliberately different from `_worst_error_metres`: ground truth is not available at the rig,
    so this is the only error an operator can ever see. That is exactly why a four-point fit is a
    trap -- it drives this to zero by construction, whatever the readings were.
    """
    projected = project_pixels_to_utm(
        homography, np.array([point.pixel_position for point in points], dtype=np.float64)
    )
    declared = np.array([point.utm_position for point in points], dtype=np.float64)
    return float(np.max(np.hypot(*(projected - declared).T)))


#: Deterministic per-point pixel noise, a couple of pixels either way -- the order of magnitude a
#: real ArUco centre wobbles by. Fixed rather than random so a failure is reproducible.
_NOISE = [
    (1.8, -1.2),
    (-1.5, 1.9),
    (1.1, 1.4),
    (-1.9, -1.6),
    (1.3, -1.8),
    (-1.2, 1.5),
    (1.7, 1.1),
    (-1.6, -1.3),
    (1.4, -1.5),
]


def test_the_homography_accepts_the_nine_point_grid_unchanged():
    """Step 5 touches no Python: this is the whole compatibility claim, stated once."""
    points = [_calibration_point(pixel) for pixel in _MEASURED_CORNERS + _GRID_EXTRAS]

    homography = create_basemap_homography(points)

    assert homography.matrix.shape == (3, 3)


def test_four_points_fit_themselves_perfectly_and_prove_nothing():
    """The trap the four-marker calibration sits in.

    With exactly four correspondences the solve has no freedom left: it passes through all four
    exactly, *including* their noise. The residual at the calibration points is therefore zero no
    matter how bad the readings were -- there is nothing to look at, and nothing to warn anyone.
    """
    points = [_calibration_point(pixel, noise) for pixel, noise in zip(_MEASURED_CORNERS, _NOISE)]

    homography = create_basemap_homography(points)

    residual = _worst_reprojection_residual_metres(homography, points)
    # Not bit-exact zero: the correspondences make a WGS84 round trip and the DLT solve is
    # float64. A tenth of a millimetre is four orders of magnitude below the millimetre-scale
    # signal this whole exercise is about, so it reads as "no residual at all".
    assert residual < 1e-4


def test_nine_points_leave_a_residual_at_the_calibration_points_themselves():
    """...and nine do not, which is the point: now there is an error to measure."""
    pixels = _MEASURED_CORNERS + _GRID_EXTRAS
    points = [_calibration_point(pixel, noise) for pixel, noise in zip(pixels, _NOISE)]

    homography = create_basemap_homography(points)

    # Orders of magnitude above the numerical floor the four-point fit sits at: the noise is
    # being spread across the solve instead of absorbed exactly.
    assert _worst_reprojection_residual_metres(homography, points) > 0.01


def test_spreading_the_points_shrinks_the_error_at_the_table_s_edges():
    """The claim step 5 rests on, measured where it matters: outside the four-marker quad.

    The corners of the *image* are well outside the quad the four markers span, so under a
    four-point fit they are pure extrapolation off a solve that absorbed all its noise. Adding
    the grid averages that noise out instead.
    """
    edges = [
        (20.0, 20.0),
        (_TABLE_WIDTH_PX - 20.0, 20.0),
        (20.0, _TABLE_HEIGHT_PX - 20.0),
        (_TABLE_WIDTH_PX - 20.0, _TABLE_HEIGHT_PX - 20.0),
    ]
    four = create_basemap_homography(
        [_calibration_point(pixel, noise) for pixel, noise in zip(_MEASURED_CORNERS, _NOISE)]
    )
    nine_pixels = _MEASURED_CORNERS + _GRID_EXTRAS
    nine = create_basemap_homography(
        [_calibration_point(pixel, noise) for pixel, noise in zip(nine_pixels, _NOISE)]
    )

    assert _worst_error_metres(nine, edges) < _worst_error_metres(four, edges)


def test_a_calibration_still_works_from_the_four_corners_alone():
    """A grid marker landing on a stitching seam must cost a correspondence, not the session."""
    homography = create_basemap_homography([_calibration_point(pixel) for pixel in _MEASURED_CORNERS])

    assert _worst_error_metres(homography, _MEASURED_CORNERS) == pytest.approx(0.0, abs=1e-4)


def test_fewer_than_four_points_is_still_refused():
    with pytest.raises(ValueError, match="at least 4"):
        create_basemap_homography([_calibration_point(pixel) for pixel in _MEASURED_CORNERS[:3]])
