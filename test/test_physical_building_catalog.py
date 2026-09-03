import math
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from build_physical_building_catalog import confirm_replace, ensure_markers_are_unique
from physical_building_catalog import (
    building_feature,
    catalog_entry,
    geometry_bbox,
    load_catalog,
    marker_index,
    normalize_geometry,
)


def feature(geometry_type="Polygon"):
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
        "properties": {"building_id": "G17", "city_scope_id": "B-17", "height": 12},
        "geometry": {"type": geometry_type, "coordinates": coordinates},
    }


@pytest.mark.parametrize("geometry_type", ["Polygon", "MultiPolygon"])
def test_normalizes_polygon_and_multipolygon_around_bbox_center(geometry_type):
    geometry, local_bbox = normalize_geometry(feature(geometry_type)["geometry"])

    assert geometry["type"] == geometry_type
    assert "local_coordinates" in geometry
    assert "coordinates" not in geometry
    assert local_bbox[0] == pytest.approx(-local_bbox[2], abs=0.02)
    assert local_bbox[1] == pytest.approx(-local_bbox[3], abs=0.02)


def test_catalog_entry_preserves_source_identity_and_units_contract():
    entry = catalog_entry(feature(), [24])

    assert entry["building_id"] == "G17"
    assert entry["city_scope_id"] == "B-17"
    assert entry["marker_ids"] == [24]
    assert entry["marker_reference_rotations"] == {"24": 0.0}
    assert entry["source_properties"]["height"] == 12


def test_empty_catalog_file_starts_a_new_catalog(tmp_path):
    path = tmp_path / "physical-building-catalog.json"
    path.touch()

    catalog = load_catalog(path)

    assert catalog["version"] == 2
    assert catalog["buildings"] == []


def test_replace_defaults_to_no():
    messages = []
    assert not confirm_replace("G17", [24], [31], lambda _prompt: "", messages.append)
    assert messages == [
        "G17 already exists with marker IDs [24].",
        "Detected marker IDs: [31].",
    ]


def test_marker_cannot_belong_to_two_buildings():
    catalog = {
        "buildings": [
            {"building_id": "G17", "marker_ids": [24]},
            {"building_id": "G18", "marker_ids": [31]},
        ]
    }
    with pytest.raises(ValueError, match="G17"):
        ensure_markers_are_unique(catalog, "G18", [24])


@pytest.mark.parametrize(
    ("center", "rotation"),
    [((10.0, 53.0), 0), ((-73.98, 40.75), 90), ((151.2, -33.87), 237)],
)
def test_runtime_geometry_is_geographic_for_different_aois_and_rotations(center, rotation):
    entry = catalog_entry(feature(), [24])
    result = building_feature(entry, 24, center, rotation)
    bbox = result["properties"]["bbox"]

    assert result["properties"]["center"] == list(center)
    assert result["properties"]["building_id"] == "G17"
    assert geometry_bbox(result["geometry"]) == bbox
    assert bbox[0] < center[0] < bbox[2]
    assert bbox[1] < center[1] < bbox[3]


def test_runtime_rotation_is_relative_to_catalogued_marker_reference():
    entry = catalog_entry(feature(), [24], {24: 170.0})
    result = building_feature(entry, 24, (10.0, 53.0), -170.0)

    assert result["properties"]["rotation"] == pytest.approx(20.0)


def test_g17_marker_24_end_to_end():
    catalog = load_catalog(Path(__file__).resolve().parents[1] / "physical-building-catalog.json")
    entry = marker_index(catalog)[24]
    result = building_feature(entry, 24, (9.99, 53.55), 32)

    assert result["properties"] == {
        "marker_id": 24,
        "building_id": "G17",
        "city_scope_id": "B-17",
        "center": [9.99, 53.55],
        "rotation": 32,
        "bbox": result["properties"]["bbox"],
    }
    assert result["geometry"]["type"] == "Polygon"
    assert "coordinates" in result["geometry"]
    assert "local_coordinates" not in result["geometry"]
