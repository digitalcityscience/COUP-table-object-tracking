"""What `build.py` tells the operator at the moment it registers a building.

D1's third leg. The first two make the gap *readable* -- `rotation_offset_deg` is `None` until
measured, and the runtime feature carries `alignment_verified: false`. Neither helps if nobody
looks. The operator is standing at the table with the block in their hand exactly once, at
registration, which is the only moment the measurement is free.
"""

import importlib.util
from pathlib import Path
import sys

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from physical_building_catalog import (
    alignment_is_verified,
    apply_building_calibration,
    catalog_entry,
)


def _load_build_module():
    """`building_catalog/build.py` by path -- it is a script beside a data folder, not a package."""
    spec = importlib.util.spec_from_file_location(
        "catalog_build", _PROJECT_ROOT / "building_catalog" / "build.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = _load_build_module()


def _entry(**calibration):
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
    entry = catalog_entry(feature, [18], {18: -116.255})
    return apply_building_calibration(entry, calibration) if calibration else entry


def test_a_just_registered_building_is_unaligned_and_therefore_gets_the_notice():
    """The condition the notice hangs off, stated as the thing it actually means."""
    assert alignment_is_verified(_entry()) is False


def test_the_notice_names_the_building_and_the_measurement_to_take():
    printed = []
    build.print_alignment_not_measured("G11", print_fn=printed.append)

    (message,) = printed
    assert "G11" in message
    assert "NOT measured" in message
    assert "rotation_offset_deg" in message


def test_the_notice_says_the_building_still_draws():
    """Registration is not being refused; the point is that it is being qualified."""
    printed = []
    build.print_alignment_not_measured("G07", print_fn=printed.append)

    assert "will draw" in printed[0]


@pytest.mark.parametrize("building_id", ["G07", "G11", "G17"])
def test_the_notice_is_specific_to_the_building_being_registered(building_id):
    printed = []
    build.print_alignment_not_measured(building_id, print_fn=printed.append)

    assert printed[0].count(building_id) >= 2
