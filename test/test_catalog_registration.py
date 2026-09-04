"""What `building_catalog/build.py` is, now that registration has left it.

Registration used to live here and it was the source of the alignment bug: a terminal tool
cannot show an operator which way the real building faces, and the ArUco angle alone cannot
supply it, so this tool wrote down whatever heading the block happened to be lying at and the
system took that as truth. It was removed rather than left beside the frontend flow, because the
two produced indistinguishable catalog entries and only one of them was measured.
"""

import importlib.util
from pathlib import Path
import sys

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))


def _load_build_module():
    """`building_catalog/build.py` by path -- it is a script beside a data folder, not a package."""
    spec = importlib.util.spec_from_file_location(
        "catalog_build", _PROJECT_ROOT / "building_catalog" / "build.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = _load_build_module()


def test_the_interactive_registration_mode_is_gone():
    """The whole point of removing it: an unmeasured registration must be unreachable.

    Left alongside the frontend flow it would keep producing catalog entries indistinguishable
    from measured ones -- which is the conflation the 2026-09-04 work exists to end.
    """
    for removed in ("build_catalog", "observe_marker_reference_rotations", "confirm_replace"):
        assert not hasattr(build, removed), f"{removed} still reachable from build.py"


def test_it_does_not_drag_in_the_camera_stack_any_more():
    """A QGIS export tool that needs a RealSense present is a tool nobody can run off-rig."""
    source = (_PROJECT_ROOT / "building_catalog" / "build.py").read_text(encoding="utf-8")

    for camera_import in ("import cv2", "camera_stitching", "detect_markers", "calibration_handler"):
        assert camera_import not in source


def test_running_it_bare_says_where_registration_went():
    """The muscle memory being interrupted is "run build.py to register a building"."""
    notice = build.REGISTRATION_MOVED_NOTICE

    assert "no longer done here" in notice
    assert "Register building" in notice
    assert "parallel" in notice


def test_the_export_mode_still_works(tmp_path):
    """The reason the file still exists at all."""
    from physical_building_catalog import catalog_entry, empty_catalog, save_catalog

    feature = {
        "type": "Feature",
        "properties": {"building_id": "G11", "city_scope_id": "B-11"},
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [[10.0, 53.0], [10.0002, 53.0], [10.0002, 53.0001], [10.0, 53.0001], [10.0, 53.0]]
            ],
        },
    }
    catalog_path = tmp_path / "catalog.json"
    catalog = empty_catalog()
    catalog["buildings"].append(catalog_entry(feature, [18], {18: -116.25}))
    save_catalog(catalog_path, catalog)

    source_path = tmp_path / "official.geojson"
    source_path.write_text(
        '{"type": "FeatureCollection", "features": [%s]}'
        % __import__("json").dumps(feature),
        encoding="utf-8",
    )
    export_path = tmp_path / "export.geojson"

    build.export_coordinate_comparison(catalog_path, source_path, export_path)

    exported = __import__("json").loads(export_path.read_text(encoding="utf-8"))
    assert [f["properties"]["building_id"] for f in exported["features"]] == ["G11"]
    assert exported["features"][0]["properties"]["marker_ids"] == [18]


def test_the_export_refuses_a_catalog_building_the_official_source_lacks(tmp_path):
    """Silently dropping it would make the comparison file quietly incomplete."""
    import json as json_module

    from physical_building_catalog import catalog_entry, empty_catalog, save_catalog

    def feature(building_id, city_scope_id):
        return {
            "type": "Feature",
            "properties": {"building_id": building_id, "city_scope_id": city_scope_id},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [[10.0, 53.0], [10.0002, 53.0], [10.0002, 53.0001], [10.0, 53.0001], [10.0, 53.0]]
                ],
            },
        }

    catalog_path = tmp_path / "catalog.json"
    catalog = empty_catalog()
    catalog["buildings"].append(catalog_entry(feature("G11", "B-11"), [18], {18: 0.0}))
    catalog["buildings"].append(catalog_entry(feature("G07", "B-07"), [12], {12: 0.0}))
    save_catalog(catalog_path, catalog)

    # The official source knows G11 but not G07.
    source_path = tmp_path / "official.geojson"
    source_path.write_text(
        json_module.dumps({"type": "FeatureCollection", "features": [feature("G11", "B-11")]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing from"):
        build.export_coordinate_comparison(catalog_path, source_path, tmp_path / "out.geojson")
