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
    building_calibration_of,
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


def test_a_just_registered_building_is_unaligned_by_construction():
    """`catalog_entry` emits no calibration block, so a fresh entry can only ever be unaligned.

    Which is why the notice is unconditional: a guard on `alignment_is_verified(entry)` here
    would be a tautology dressed up as a check.
    """
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


# --- re-registration throws a measured alignment away, and must say so -------------------


def test_re_registering_reports_the_calibration_it_just_invalidated():
    """The same conflation, one level up: a destroyed measurement must not read as "never made".

    `rotation_offset_deg` is a residual against the *stored reference heading*. Re-registration
    captures a new reference, so the old offset describes a heading that no longer exists --
    carrying it over would be worse than dropping it. Dropping it silently, though, leaves the
    generic "not measured" banner reading identically whether this is a first registration or an
    operator who has just lost a sitting's work.
    """
    printed = []
    build.print_alignment_discarded(
        "G17", building_calibration_of(_entry(rotation_offset_deg=-47.3, offset_east_m=0.35)), printed.append
    )

    (message,) = printed
    assert "G17" in message
    assert "-47.300" in message
    assert "discarded" in message


def test_a_building_that_was_never_aligned_has_nothing_to_report_as_discarded():
    """The condition the caller guards on: only a *measured* offset is worth warning about."""
    assert building_calibration_of(_entry())["rotation_offset_deg"] is None
    assert building_calibration_of(_entry(offset_east_m=0.35))["rotation_offset_deg"] is None


def test_a_measured_zero_still_counts_as_something_to_lose():
    """Somebody stood at the table and established this. Losing it silently is losing work."""
    assert building_calibration_of(_entry(rotation_offset_deg=0.0))["rotation_offset_deg"] == 0.0
