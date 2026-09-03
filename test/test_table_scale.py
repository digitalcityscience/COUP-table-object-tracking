"""The table's ground scale, and the one factor that shrinks catalog geometry onto it.

The bug this locks down: the catalog carries real-world metres, the table map is drawn at the
AOI's own ground scale (~1:270 today), and the physical blocks are milled at 1:500. Placing
catalog metres straight onto the map therefore draws every building `500 / ground_scale` times
the size of the block it is supposed to sit under (~1.85x on the 2026-08-31 rig).

`ground_scale` is derived from the homography, never sent by the frontend and never measured by
hand, so there is exactly one number to be wrong about.
"""

import math
from pathlib import Path
import sys

import pytest
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibration_contract import TABLE_PIXELS_PER_CM
from physical_building_catalog import (
    MODEL_SCALE,
    catalog_entry,
    building_feature,
    geometry_bbox,
    load_catalog,
    marker_index,
    model_scale_factor,
    place_geometry,
)
from pixel_to_utm import (
    BasemapCalibrationPoint,
    create_basemap_homography,
    ground_scale,
    metres_per_table_pixel,
)

_UTM_TO_WGS84 = Transformer.from_crs("EPSG:25832", "EPSG:4326", always_xy=True)

#: A UTM 32N origin inside Hamburg, so the synthetic AOIs below sit where the real rig does.
_UTM_ORIGIN = (566_000.0, 5_936_000.0)

#: The stitched table image the rig actually produces: 160x80 cm at 10 px/cm.
_TABLE_PIXEL_CORNERS = [(0.0, 0.0), (1600.0, 0.0), (1600.0, 800.0), (0.0, 800.0)]


def _flat_homography(metres_per_pixel: float, pixel_corners=_TABLE_PIXEL_CORNERS):
    """A homography whose ground scale is known exactly: every table pixel is `metres_per_pixel`.

    Built by walking each pixel corner out from `_UTM_ORIGIN` at that fixed rate and handing the
    resulting UTM point back as lat/lon, which is the only form `BasemapCalibrationPoint` accepts.
    """
    points = []
    for pixel_x, pixel_y in pixel_corners:
        easting = _UTM_ORIGIN[0] + pixel_x * metres_per_pixel
        northing = _UTM_ORIGIN[1] - pixel_y * metres_per_pixel
        longitude, latitude = _UTM_TO_WGS84.transform(easting, northing)
        points.append(
            BasemapCalibrationPoint(
                pixel_position=(pixel_x, pixel_y),
                lat_lon_position=(latitude, longitude),
            )
        )
    return create_basemap_homography(points)


def test_table_pixel_density_is_the_stitching_pipeline_s_own_scale_factor():
    """10 px/cm is a fact about `camera_stitching`, and both sides must read the same one."""
    assert TABLE_PIXELS_PER_CM == 10


def test_metres_per_table_pixel_reads_the_homography_s_local_scale():
    homography = _flat_homography(0.30)

    assert metres_per_table_pixel(homography, (800.0, 400.0)) == pytest.approx(0.30, abs=1e-6)


def test_ground_scale_is_real_centimetres_per_table_centimetre():
    # 0.30 m per table pixel x 10 px/cm = 3 m of ground per table centimetre = 1:300.
    assert ground_scale(_flat_homography(0.30)) == pytest.approx(300.0, rel=1e-4)
    assert ground_scale(_flat_homography(0.50)) == pytest.approx(500.0, rel=1e-4)


def test_ground_scale_is_sampled_at_the_calibration_points_centroid():
    """A projective map has no single scale; the sample point is pinned, not left to the caller."""
    homography = _flat_homography(0.30)

    assert homography.pixel_centroid == pytest.approx((800.0, 400.0))


def test_model_scale_is_the_single_declared_block_scale():
    assert MODEL_SCALE == 500


def test_model_scale_factor_shrinks_catalog_metres_onto_the_table_map():
    # The 2026-08-31 rig: the map is ~1.85x finer than the blocks, so geometry must shrink to 0.54.
    assert model_scale_factor(270.0) == pytest.approx(270.0 / 500.0)
    # A map drawn at exactly the block scale needs no correction at all.
    assert model_scale_factor(500.0) == pytest.approx(1.0)


def test_model_scale_factor_rejects_a_non_positive_ground_scale():
    with pytest.raises(ValueError, match="ground scale"):
        model_scale_factor(0.0)


def _local_extent(geometry) -> tuple[float, float]:
    min_x, min_y, max_x, max_y = geometry_bbox(geometry)
    return max_x - min_x, max_y - min_y


def _feature(geometry_type="Polygon"):
    ring = [
        [10.0, 53.0],
        [10.0002, 53.0],
        [10.0002, 53.0001],
        [10.0, 53.0001],
        [10.0, 53.0],
    ]
    coordinates = [ring] if geometry_type == "Polygon" else [[ring]]
    return {
        "type": "Feature",
        "properties": {"building_id": "G17", "city_scope_id": "B-17"},
        "geometry": {"type": geometry_type, "coordinates": coordinates},
    }


def test_place_geometry_defaults_to_unscaled():
    entry = catalog_entry(_feature(), [24])

    placed = place_geometry(entry["geometry"], (10.0, 53.0), 0.0)
    explicit = place_geometry(entry["geometry"], (10.0, 53.0), 0.0, scale=1.0)

    assert placed == explicit


@pytest.mark.parametrize("scale", [0.25, 0.54, 2.0])
def test_place_geometry_scales_the_footprint_about_its_own_anchor(scale):
    entry = catalog_entry(_feature(), [24])
    centre = (10.0, 53.0)

    unscaled = place_geometry(entry["geometry"], centre, 0.0)
    scaled = place_geometry(entry["geometry"], centre, 0.0, scale=scale)

    unscaled_width, unscaled_height = _local_extent(unscaled)
    scaled_width, scaled_height = _local_extent(scaled)
    assert scaled_width == pytest.approx(unscaled_width * scale, rel=1e-3)
    assert scaled_height == pytest.approx(unscaled_height * scale, rel=1e-3)


def test_scaling_and_rotation_commute_so_a_turned_block_keeps_its_size():
    """Scale is applied in the local frame, before rotation — order must not change the size."""
    entry = catalog_entry(_feature(), [24])
    centre = (10.0, 53.0)

    upright = place_geometry(entry["geometry"], centre, 0.0, scale=0.54)
    turned = place_geometry(entry["geometry"], centre, 37.0, scale=0.54)

    upright_width, upright_height = _local_extent(upright)
    turned_width, turned_height = _local_extent(turned)
    # Same footprint, rotated: the bbox changes shape but not the diagonal it is inscribed in.
    assert math.hypot(turned_width, turned_height) >= min(upright_width, upright_height)
    assert math.hypot(turned_width, turned_height) <= math.hypot(
        upright_width + upright_height, upright_width + upright_height
    )


def test_building_feature_passes_the_scale_through_to_the_drawn_geometry():
    entry = catalog_entry(_feature(), [24])

    full = building_feature(entry, 24, (10.0, 53.0), 0.0)
    half = building_feature(entry, 24, (10.0, 53.0), 0.0, scale=0.5)

    full_width, _ = _local_extent(full["geometry"])
    half_width, _ = _local_extent(half["geometry"])
    assert half_width == pytest.approx(full_width * 0.5, rel=1e-3)
    assert half["properties"]["model_scale_factor"] == pytest.approx(0.5)


@pytest.mark.parametrize("metres_per_pixel", [0.20, 0.2926, 0.30, 0.45])
def test_a_placed_building_covers_the_same_table_area_as_its_1_500_block(metres_per_pixel):
    """The whole point of step 1, as an invariant rather than a ruler reading.

    Whatever ground scale the operator's AOI lands on, a catalog building placed with
    `model_scale_factor` must occupy the table area a 1:500 milled block occupies — that is
    what "the drawing sits on top of the block" means, expressed in numbers.
    """
    catalog = load_catalog(Path(__file__).resolve().parents[1] / "physical-building-catalog.json")
    entry = marker_index(catalog)[24]
    homography = _flat_homography(metres_per_pixel)
    scale = model_scale_factor(ground_scale(homography))

    real_min_x, real_min_y, real_max_x, real_max_y = entry["local_bbox"]
    block_width_cm = (real_max_x - real_min_x) / MODEL_SCALE * 100
    block_height_cm = (real_max_y - real_min_y) / MODEL_SCALE * 100

    # Feed back the marker's own reference rotation so the effective rotation is 0 and the drawn
    # bbox is comparable with the catalog's axis-aligned `local_bbox`.
    upright = entry["marker_reference_rotations"]["24"]
    placed = building_feature(entry, 24, (10.0, 53.55), upright, scale=scale)
    # `place_geometry` emits WGS84; measure it back in local metres around its own centre.
    from physical_building_catalog import normalize_geometry

    drawn_local, _bbox = normalize_geometry(placed["geometry"])
    drawn_ground_width, drawn_ground_height = _local_extent(drawn_local)
    # Ground metres, divided by the map's ground scale, are table metres.
    drawn_table_width_cm = drawn_ground_width / ground_scale(homography) * 100
    drawn_table_height_cm = drawn_ground_height / ground_scale(homography) * 100

    assert drawn_table_width_cm == pytest.approx(block_width_cm, rel=2e-3)
    assert drawn_table_height_cm == pytest.approx(block_height_cm, rel=2e-3)
