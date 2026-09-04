"""The two things a table-to-ground map carries besides position: direction, and per-axis scale.

Both were being read off the homography as if it were a similarity transform -- a rotation plus
one uniform scale. It is not. The 2026-09-04 rig measured 0.420 degrees of axis-squareness error
(shear) and |dx|/|dy| = 0.9641 (3.5% anisotropy), which breaks the two assumptions separately:

- D2: an angle copied across as a bare scalar is wrong by up to 1.8 degrees, varying with heading
  *and* position, so no constant `rotation_offset_deg` can absorb it.
- D3: `metres_per_table_pixel` probed only the pixel-x axis, on the stated grounds that the two
  axes carry the same scale. They differ by 3.5%, so `ground_scale` and every building's
  `model_scale_factor` were systematically wrong.

The homographies below are built from an explicit 2x2 matrix so the right answer is known in
closed form rather than measured from the thing under test.
"""

import math
from pathlib import Path
import sys

import pytest
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pixel_to_utm import (
    BasemapCalibrationPoint,
    axis_metres_per_table_pixel,
    create_basemap_homography,
    direction_through_homography,
    ground_scale,
    metres_per_table_pixel,
)

_UTM_TO_WGS84 = Transformer.from_crs("EPSG:25832", "EPSG:4326", always_xy=True)

#: A UTM 32N origin inside Hamburg, so the synthetic AOIs sit where the real rig does.
_UTM_ORIGIN = (566_000.0, 5_936_000.0)

#: The stitched table image the rig actually produces: 160x80 cm at 10 px/cm.
_TABLE_PIXEL_CORNERS = [(0.0, 0.0), (1600.0, 0.0), (1600.0, 800.0), (0.0, 800.0)]

#: The rig's own numbers (`calibration_20260904_091415`), as a pixel -> UTM matrix: the +x pixel
#: axis carries 0.3297 m/px and points 0.420 degrees off square from the +y axis, which carries
#: 0.3420 m/px. Columns are where each pixel axis lands in (east, north) metres.
_RIG_MATRIX = (
    (0.3297 * math.cos(math.radians(0.420)), 0.3297 * math.sin(math.radians(0.420))),
    (0.0, 0.3420),
)


def _linear_homography(matrix, pixel_corners=_TABLE_PIXEL_CORNERS):
    """A homography realising `matrix` exactly: pixel (x, y) -> _UTM_ORIGIN + matrix @ (x, y).

    Written column-wise -- `matrix[0]` is where one pixel of +x lands in (east, north) metres and
    `matrix[1]` is where one pixel of +y lands -- because that is the form the expectations below
    are stated in, and it keeps a transposed matrix from quietly passing.
    """
    (east_x, north_x), (east_y, north_y) = matrix
    points = []
    for pixel_x, pixel_y in pixel_corners:
        easting = _UTM_ORIGIN[0] + pixel_x * east_x + pixel_y * east_y
        northing = _UTM_ORIGIN[1] + pixel_x * north_x + pixel_y * north_y
        longitude, latitude = _UTM_TO_WGS84.transform(easting, northing)
        points.append(
            BasemapCalibrationPoint(
                pixel_position=(pixel_x, pixel_y),
                lat_lon_position=(latitude, longitude),
            )
        )
    return create_basemap_homography(points)


def _expected_direction(matrix, table_degrees: float) -> float:
    (east_x, north_x), (east_y, north_y) = matrix
    cosine, sine = math.cos(math.radians(table_degrees)), math.sin(math.radians(table_degrees))
    return math.degrees(
        math.atan2(north_x * cosine + north_y * sine, east_x * cosine + east_y * sine)
    )


_CENTRE = (800.0, 400.0)


# --- D2: direction ----------------------------------------------------------------------


@pytest.mark.parametrize("table_degrees", [0.0, 30.0, 45.0, 90.0, 135.0, -120.0, 179.0])
def test_a_direction_is_whatever_the_homography_says_it_is(table_degrees):
    """The closed-form answer for the rig's own matrix, at every heading a block can take."""
    homography = _linear_homography(_RIG_MATRIX)

    assert direction_through_homography(
        homography, _CENTRE, table_degrees
    ) == pytest.approx(_expected_direction(_RIG_MATRIX, table_degrees), abs=1e-3)


def test_a_similarity_map_shifts_every_direction_by_the_same_constant():
    """The control case: when the map really is a rotation plus one scale, a scalar would do.

    This is the assumption the old scalar path was built on. It is not wrong in general -- it is
    wrong about *this* rig, which is what the next test pins down.
    """
    turn = math.radians(20.0)
    similarity = (
        (0.33 * math.cos(turn), 0.33 * math.sin(turn)),
        (-0.33 * math.sin(turn), 0.33 * math.cos(turn)),
    )
    homography = _linear_homography(similarity)

    shifts = [
        (direction_through_homography(homography, _CENTRE, degrees) - degrees + 180.0) % 360.0
        - 180.0
        for degrees in (0.0, 45.0, 90.0, 135.0)
    ]

    assert shifts == pytest.approx([shifts[0]] * 4, abs=1e-3)


def test_the_rig_s_shear_makes_the_shift_depend_on_heading():
    """Why a scalar angle cannot be saved by a constant `rotation_offset_deg`.

    The spread across headings is the part no per-building constant can absorb. The doc measured
    a 2.576 degree peak-to-peak across headings and positions; the shear alone, at the table
    centre, is most of it.
    """
    homography = _linear_homography(_RIG_MATRIX)

    shifts = [
        (direction_through_homography(homography, _CENTRE, degrees) - degrees + 180.0) % 360.0
        - 180.0
        for degrees in range(0, 180, 5)
    ]

    assert max(shifts) - min(shifts) > 0.9


def test_the_shift_is_180_degree_periodic():
    """The signature of shear plus anisotropy: a linear map takes v and -v to opposite rays."""
    homography = _linear_homography(_RIG_MATRIX)

    for degrees in (0.0, 37.0, 90.0, 128.0):
        forward = direction_through_homography(homography, _CENTRE, degrees)
        backward = direction_through_homography(homography, _CENTRE, degrees + 180.0)
        assert (backward - forward + 360.0) % 360.0 == pytest.approx(180.0, abs=1e-3)


def test_a_direction_keeps_its_sign():
    """The `-1` that started all of this: turning the block left must read as a bigger angle."""
    homography = _linear_homography(_RIG_MATRIX)

    baseline = direction_through_homography(homography, _CENTRE, 0.0)
    turned = direction_through_homography(homography, _CENTRE, 90.0)

    assert (turned - baseline + 360.0) % 360.0 == pytest.approx(90.0, abs=1.0)


def test_the_probe_length_does_not_change_the_answer_on_a_linear_map():
    """Guards the finite difference: a longer secant must not drag in a different part of the map."""
    homography = _linear_homography(_RIG_MATRIX)

    assert direction_through_homography(
        homography, _CENTRE, 33.0, probe_pixels=1.0
    ) == pytest.approx(
        direction_through_homography(homography, _CENTRE, 33.0, probe_pixels=100.0), abs=1e-3
    )


# --- D3: per-axis scale -----------------------------------------------------------------


def test_the_two_axis_scales_are_read_separately():
    homography = _linear_homography(_RIG_MATRIX)

    east_metres, north_metres = axis_metres_per_table_pixel(homography, _CENTRE)

    assert east_metres == pytest.approx(0.3297, rel=1e-4)
    assert north_metres == pytest.approx(0.3420, rel=1e-4)


def test_the_rig_really_is_anisotropic():
    """The docstring claim D3 retired: "the two axes carry the same scale". They do not."""
    east_metres, north_metres = axis_metres_per_table_pixel(
        _linear_homography(_RIG_MATRIX), _CENTRE
    )

    assert east_metres / north_metres == pytest.approx(0.9641, abs=1e-3)


def test_the_scale_is_the_geometric_mean_of_both_axes():
    """Not the x axis alone, which was ~1.8% low against the mean and 3.6% against the y axis."""
    homography = _linear_homography(_RIG_MATRIX)

    assert metres_per_table_pixel(homography, _CENTRE) == pytest.approx(
        math.sqrt(0.3297 * 0.3420), rel=1e-4
    )


def test_the_geometric_mean_preserves_the_area_a_drawing_covers():
    """Why geometric and not arithmetic: one isotropic factor for a two-dimensional footprint."""
    homography = _linear_homography(_RIG_MATRIX)
    east_metres, north_metres = axis_metres_per_table_pixel(homography, _CENTRE)

    single = metres_per_table_pixel(homography, _CENTRE)

    assert single**2 == pytest.approx(east_metres * north_metres, rel=1e-6)


def test_an_isotropic_map_is_unaffected_by_the_two_axis_probe():
    """The fix must be a no-op wherever the old single-axis assumption actually held."""
    isotropic = ((0.30, 0.0), (0.0, 0.30))

    assert metres_per_table_pixel(_linear_homography(isotropic), _CENTRE) == pytest.approx(
        0.30, abs=1e-6
    )
    assert ground_scale(_linear_homography(isotropic)) == pytest.approx(300.0, rel=1e-4)


# --- D2 leaves the already-registered references valid ----------------------------------


def test_a_block_at_its_reference_heading_still_draws_at_zero_rotation():
    """The property that makes D2 a pure fix instead of a convention change.

    `build.py` records `marker_reference_rotations` in raw table-pixel degrees; it has no
    homography. Running the conversion over the detected angle *only* would subtract two numbers
    from different frames and force every registered reference to be migrated arithmetically --
    which is what the design note expected to have to do. Running it over both, at the marker's
    own pixel, makes the conversion cancel exactly when the block is at its reference heading, at
    any position on the table. So the three registered buildings keep the references they have.

    This is also the invariant D1's measurement table rests on: `detected == reference` must land
    on the catalog's own heading to 0.000 degrees, or the alignment procedure measures nothing.
    """
    from physical_building_catalog import building_feature, load_catalog, marker_index
    from functools import partial

    homography = _linear_homography(_RIG_MATRIX)
    catalog = load_catalog(Path(__file__).resolve().parents[1] / "physical-building-catalog.json")

    for marker_id, building in marker_index(catalog).items():
        reference = float(building["marker_reference_rotations"][str(marker_id)])
        for pixel in [(200.0, 150.0), (800.0, 400.0), (1400.0, 650.0)]:
            drawn = building_feature(
                building,
                marker_id,
                (9.99, 53.55),
                reference,
                table_direction_to_map=partial(direction_through_homography, homography, pixel),
            )
            assert drawn["properties"]["rotation"] == pytest.approx(0.0, abs=1e-9)
