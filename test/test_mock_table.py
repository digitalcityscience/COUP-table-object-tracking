"""What the mock table has to keep true to be worth developing a frontend against.

The mock's value rests entirely on one claim: a frontend that works against it works against the
rig. These tests pin the parts of that claim the mock could break on its own -- the snapshot's
shape, the marker ids, and the geometric round-trip -- rather than re-testing `server`, which the
mock runs unmodified and which has its own suite.
"""

import math

import pytest
from pyproj import Transformer

import mock_table
from calibration_contract import MAP_CALIBRATION_MARKER_CORNERS, MAP_CALIBRATION_MARKER_IDS
from mock_table import CALIBRATION_MARKER_BANDS, MockTable
from pixel_to_utm import BasemapCalibrationPoint, create_basemap_homography, project_pixel_to_utm

_UTM_TO_WGS84 = Transformer.from_crs("EPSG:25832", "EPSG:4326", always_xy=True)

#: An AOI the size of the frontend's overlay, somewhere over HafenCity. Any rectangle would do --
#: the round-trip below is scale-free -- but a real one keeps the UTM numbers in the range the
#: homography actually solves at.
AOI_CENTER_LNG, AOI_CENTER_LAT = 10.0, 53.54
AOI_WIDTH_M, AOI_HEIGHT_M = 800.0, 400.0


@pytest.fixture
def table():
    import server

    return MockTable(server.physical_building_catalog, motion="still", seed=1)


def _aoi_lat_lon(u: float, v: float) -> tuple[float, float]:
    """The `(lat, lon)` at table fraction `(u, v)`, in the order `map_calibration` wants it.

    `lat_lon_position` really is latitude first (`pixel_to_utm.BasemapCalibrationPoint.
    latlon_to_utm`), unlike `register_building`'s `target`, which is `[lng, lat]`. Getting this
    backwards produces a homography that fits its own points perfectly and puts every building in
    the wrong place, so it is spelled out here rather than inlined.
    """
    metres_per_deg_lat = 111320.0
    metres_per_deg_lng = 111320.0 * math.cos(math.radians(AOI_CENTER_LAT))
    east = (u - 0.5) * AOI_WIDTH_M
    north = (0.5 - v) * AOI_HEIGHT_M
    return (
        AOI_CENTER_LAT + north / metres_per_deg_lat,
        AOI_CENTER_LNG + east / metres_per_deg_lng,
    )


def _homography_from(snapshot):
    """The homography the frontend would build from this snapshot's calibration markers."""
    min_u, max_u = mock_table.INSET_U, 1.0 - mock_table.INSET_U
    min_v, max_v = mock_table.INSET_V, 1.0 - mock_table.INSET_V
    points = [
        BasemapCalibrationPoint(
            pixel_position=tuple(snapshot[marker_id][:2]),
            lat_lon_position=_aoi_lat_lon(
                mock_table._band_fraction(column, min_u, max_u),
                mock_table._band_fraction(row, min_v, max_v),
            ),
        )
        for marker_id, column, row in CALIBRATION_MARKER_BANDS
    ]
    return create_basemap_homography(points)


def test_the_mock_emits_every_calibration_marker_the_contract_names(table):
    """A missing id is invisible: the frontend simply never calibrates, with no error anywhere."""
    snapshot = table.snapshot()
    assert set(MAP_CALIBRATION_MARKER_IDS) <= set(snapshot)


def test_a_reading_has_the_shape_marker_printJSON_produces(table):
    """`[x, y, rotation, camera_id]` with an int key -- what every consumer downstream unpacks."""
    snapshot = table.snapshot()
    for marker_id, reading in snapshot.items():
        assert isinstance(marker_id, int)
        assert len(reading) == 4
        x, y, rotation, camera_id = reading
        assert isinstance(x, float) and isinstance(y, float) and isinstance(rotation, float)
        assert camera_id == mock_table.CAMERA_ID


def test_the_three_catalogued_blocks_are_on_the_table(table):
    snapshot = table.snapshot()
    assert {12, 18, 24} <= set(snapshot)


def test_each_block_starts_at_the_heading_its_catalog_reference_was_measured_at(table):
    """The mock's rest state is an *aligned* table.

    Without this a developer cannot tell a frontend rotation bug from the mock simply having
    started the block turned, which is the one thing a simulator must never be ambiguous about.
    """
    import server

    catalog = {b["building_id"]: b for b in server.physical_building_catalog["buildings"]}
    for block in table.buildings.values():
        stored = catalog[block.building_id]["marker_reference_rotations"][str(block.marker_id)]
        assert block.heading == pytest.approx(stored)
        assert block.heading_from_aligned == pytest.approx(0.0)


def test_a_block_lands_where_the_homography_says_it_should(table):
    """The round-trip the whole mock exists for: uv -> pixel -> homography -> lng/lat -> uv.

    A mock whose pixel frame disagreed with the frontend's marker layout would still calibrate --
    `cv2.findHomography` fits whatever it is given -- and would then place every building somewhere
    plausible but wrong. This is the only test that can catch that, because it is the only one that
    checks a *building* pixel against the frame the *calibration* markers defined.
    """
    snapshot = table.snapshot()
    homography = _homography_from(snapshot)

    for block in table.buildings.values():
        pixel = tuple(snapshot[block.marker_id][:2])
        easting, northing = project_pixel_to_utm(homography, pixel)
        lng, lat = _UTM_TO_WGS84.transform(easting, northing)
        expected_lat, expected_lng = _aoi_lat_lon(block.u, block.v)
        # Sub-metre: the AOI is 800 m across, so this is well inside a tenth of a percent.
        assert lng == pytest.approx(expected_lng, abs=1e-5)
        assert lat == pytest.approx(expected_lat, abs=1e-5)


def test_the_calibration_quad_is_not_mirrored(table):
    """Pixel `y` must run opposite to north, as it does on the rig.

    A mirrored mock is the worst failure this module can have: the homography still fits its ten
    points exactly, nothing reports an error, and every building is reflected across the table.
    """
    snapshot = table.snapshot()
    top_left = snapshot[200]
    bottom_left = snapshot[202]
    assert top_left[1] > bottom_left[1], "the AOI's north edge must read at a LARGER pixel y"
    assert snapshot[201][0] > top_left[0], "east must read at a larger pixel x"


def test_taking_a_block_off_the_table_stops_it_being_detected(table):
    table.set_on_table("G11", False)
    assert 18 not in table.snapshot()
    table.set_on_table("G11", True)
    assert 18 in table.snapshot()


def test_an_unclaimed_marker_appears_with_no_catalog_entry(table):
    """Registration's whole subject: a block on the table that no building speaks for."""
    import server

    table.add_unclaimed(42)
    assert 42 in table.snapshot()
    assert server.physical_buildings_by_marker.get(42) is None


def test_a_block_can_be_named_by_building_id_or_marker_id(table):
    assert table.block("g11") is table.block("18")
    with pytest.raises(KeyError):
        table.block("G99")


def test_still_motion_is_exactly_repeatable(table):
    """`--motion still` is what makes a mock session reproducible enough to bisect against."""
    assert table.snapshot() == table.snapshot()


def test_jitter_moves_the_reading_without_moving_the_block(table):
    table.set_motion("jitter")
    first, second = table.snapshot(), table.snapshot()
    assert first != second
    for marker_id, reading in first.items():
        assert reading[0] == pytest.approx(second[marker_id][0], abs=5.0)


def test_the_corner_ids_still_mean_what_the_contract_says(table):
    """200/201/202/203 -> top_left/top_right/bottom_left/bottom_right is a physical contract."""
    assert {marker_id for marker_id, _, _ in CALIBRATION_MARKER_BANDS[:4]} == set(
        MAP_CALIBRATION_MARKER_CORNERS
    )
    snapshot = table.snapshot()
    assert snapshot[200][0] < snapshot[201][0]  # top_left is west of top_right
    assert snapshot[202][0] < snapshot[203][0]  # bottom_left is west of bottom_right
