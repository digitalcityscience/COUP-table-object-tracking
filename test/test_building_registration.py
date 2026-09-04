"""Registering a block against the building it stands for.

The thing registration has to establish is not the block's heading -- a camera reads that thirty
times a second -- but the relationship between that heading and the direction the real building
faces. That relationship is not in the ArUco angle and never can be: the same marker glued
straight, turned 90 degrees, or upside down produces the same reading. A human supplies it once,
by turning the block parallel to its own footprint projected on the table.

These tests pin down what that confirmation is allowed to produce, and -- more importantly -- what
it must refuse, because a bad reference becomes a silent constant error in every later drawing.
"""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from building_registration import (
    MAXIMUM_REFERENCE_SPREAD_DEG,
    marker_on_target,
    markers_on_the_table,
    named_marker,
    MINIMUM_REFERENCE_SAMPLES,
    angular_spread_degrees,
    catalog_with_entry,
    circular_mean_degrees,
    marker_to_register,
    reference_rotation_from_samples,
    registered_entry,
)
from physical_building_catalog import (
    alignment_is_verified,
    building_calibration_of,
    catalog_entry,
    empty_catalog,
)


def _feature(building_id="G11", city_scope_id="B-11"):
    ring = [
        [10.0, 53.0],
        [10.0002, 53.0],
        [10.0002, 53.0001],
        [10.0, 53.0001],
        [10.0, 53.0],
    ]
    return {
        "type": "Feature",
        "properties": {"building_id": building_id, "city_scope_id": city_scope_id},
        "geometry": {"type": "Polygon", "coordinates": [ring]},
    }


def _samples(*angles, count=MINIMUM_REFERENCE_SAMPLES):
    """`count` readings cycling through `angles`, as the snapshot buffer would hold them."""
    return [angles[index % len(angles)] for index in range(count)]


# --- averaging a heading that wraps ------------------------------------------------------


def test_the_mean_of_two_readings_either_side_of_the_seam_is_not_zero():
    """The bug a plain arithmetic mean would introduce, and it points the block backwards.

    179 and -179 are two degrees apart. Averaged as numbers they give 0 -- a heading 180 degrees
    from the truth, filed into the catalog as a constant, and impossible to spot afterwards
    because it looks like a perfectly ordinary reference.
    """
    assert abs(circular_mean_degrees([179.0, -179.0])) == pytest.approx(180.0, abs=1e-6)


def test_an_ordinary_run_of_readings_averages_where_you_would_expect():
    assert circular_mean_degrees([-116.0, -116.5, -116.25]) == pytest.approx(-116.25, abs=1e-3)


def test_the_spread_is_measured_across_the_seam_too():
    """Max-minus-min would call these 358 degrees apart and refuse a perfectly good block."""
    assert angular_spread_degrees([179.0, -179.0, 180.0]) == pytest.approx(1.0, abs=1e-6)


def test_an_empty_set_of_angles_is_refused_rather_than_averaged():
    with pytest.raises(ValueError, match="empty"):
        circular_mean_degrees([])


# --- what may become a reference ---------------------------------------------------------


def test_a_still_block_produces_its_heading():
    reference = reference_rotation_from_samples(_samples(-116.2, -116.3, -116.25))

    assert reference == pytest.approx(-116.25, abs=0.1)


def test_too_few_readings_are_refused():
    """Two seconds of stillness, or the mean is single-frame corner noise wearing a mean's hat."""
    with pytest.raises(ValueError, match="reading"):
        reference_rotation_from_samples([-116.2] * (MINIMUM_REFERENCE_SAMPLES - 1))


def test_a_block_that_was_still_moving_is_refused():
    """The failure this guard exists for: confirming before the block settled.

    A heading averaged across a moving block is not a heading, and filing it bakes the error in
    as a constant -- exactly the silent, permanent kind of wrongness this whole flow exists to
    end. Refusing costs the operator one more confirmation.
    """
    moving = [-116.0 + index * 5.0 for index in range(MINIMUM_REFERENCE_SAMPLES)]

    with pytest.raises(ValueError, match="moved"):
        reference_rotation_from_samples(moving)


def test_ordinary_corner_noise_is_not_mistaken_for_movement():
    """The rig's own worst measured jitter (+/-6 deg) has to pass, or nothing can be registered."""
    jittery = _samples(-110.0, -122.0, -116.0)

    assert reference_rotation_from_samples(jittery) == pytest.approx(-116.0, abs=1.0)
    assert angular_spread_degrees(jittery) < MAXIMUM_REFERENCE_SPREAD_DEG


# --- which marker is being registered ----------------------------------------------------


def _catalog_with(*buildings):
    catalog = empty_catalog()
    catalog["buildings"] = list(buildings)
    return catalog


def test_the_marker_being_registered_is_the_one_no_building_claims():
    """The frontend cannot send a marker id for a building that has no catalog entry yet."""
    known = catalog_entry(_feature("G07", "B-07"), [12], {12: 0.0})

    assert marker_to_register(_catalog_with(known), "G11", [12, 18, 200], [200]) == 18


def test_re_registering_a_building_finds_its_own_marker_even_though_it_is_claimed():
    """A re-glued marker means re-registration, and that building's marker is already in the catalog."""
    existing = catalog_entry(_feature(), [18], {18: -116.25})

    assert marker_to_register(_catalog_with(existing), "G11", [18, 200], [200]) == 18


def test_registering_with_two_unclaimed_blocks_on_the_table_is_refused():
    """Guessing would attach a building to the wrong physical block, permanently and silently."""
    with pytest.raises(ValueError, match="Leave only"):
        marker_to_register(empty_catalog(), "G11", [18, 24], [200])


def test_registering_with_no_block_on_the_table_is_refused():
    known = catalog_entry(_feature("G07", "B-07"), [12], {12: 0.0})

    with pytest.raises(ValueError, match="no unclaimed marker"):
        marker_to_register(_catalog_with(known), "G11", [12, 200], [200])


def test_calibration_markers_are_never_candidates():
    """The Table window projects them, so they are in view during every registration."""
    with pytest.raises(ValueError, match="no unclaimed marker"):
        marker_to_register(empty_catalog(), "G11", [200, 201, 202, 203], [200, 201, 202, 203])


# --- the entry the confirmation produces -------------------------------------------------


def test_a_confirmed_registration_is_aligned_and_says_so():
    """The point of the whole flow: `rotation_offset_deg` is a *measured* zero, not an absent one.

    The operator has just been shown the building's real heading and put the block on it, so the
    residual is nil and somebody checked. Leaving it `None` here would throw away the one
    measurement the procedure exists to take -- as wrong in the other direction as the old code
    writing a blind `0.0`.
    """
    entry = registered_entry(_feature(), 18, -116.25)

    assert alignment_is_verified(entry) is True
    assert building_calibration_of(entry)["rotation_offset_deg"] == pytest.approx(0.0)
    assert entry["marker_reference_rotations"] == {"18": pytest.approx(-116.25)}
    assert entry["building_id"] == "G11"


def test_the_other_calibration_fields_stay_neutral():
    """Alignment is what was measured. The offsets and scale were not, and must not be claimed."""
    stored = building_calibration_of(registered_entry(_feature(), 18, -116.25))

    assert stored["offset_east_m"] == pytest.approx(0.0)
    assert stored["offset_north_m"] == pytest.approx(0.0)
    assert stored["scale_residual"] == pytest.approx(1.0)


def test_registering_replaces_the_building_rather_than_adding_a_second_one():
    """A re-glued marker must not leave the old one still claiming the building."""
    catalog = _catalog_with(catalog_entry(_feature(), [18], {18: 0.0}))

    updated = catalog_with_entry(catalog, registered_entry(_feature(), 99, -116.25))

    assert [building["building_id"] for building in updated["buildings"]] == ["G11"]
    assert updated["buildings"][0]["marker_ids"] == [99]


def test_registering_leaves_other_buildings_untouched():
    other = catalog_entry(_feature("G07", "B-07"), [12], {12: 5.0})
    catalog = _catalog_with(other)

    updated = catalog_with_entry(catalog, registered_entry(_feature(), 18, -116.25))

    assert sorted(b["building_id"] for b in updated["buildings"]) == ["G07", "G11"]
    assert updated["buildings"][0]["marker_reference_rotations"] == {"12": pytest.approx(5.0)}


def test_registering_does_not_mutate_the_catalog_it_was_given():
    """A refused write must never have half-changed the live catalog."""
    catalog = _catalog_with(catalog_entry(_feature(), [18], {18: 0.0}))

    catalog_with_entry(catalog, registered_entry(_feature(), 99, -116.25))

    assert catalog["buildings"][0]["marker_ids"] == [18]


# --- which block is this? the operator already answered, physically ----------------------


def test_the_marker_on_the_target_is_the_one_being_registered():
    """Position answers it, so nothing else on the table has to be moved or explained."""
    on_table = {18: (700.0, 400.0), 24: (1300.0, 200.0), 12: (200.0, 650.0)}

    assert marker_on_target(on_table, (705.0, 395.0)) == 18


def test_phantom_reads_elsewhere_on_the_table_are_simply_not_on_the_target():
    """The refusal this replaces: `markers [18, 85, 182, 190] are all unclaimed`.

    A noisy frame produces spurious ArUco ids at arbitrary places. Under elimination every one of
    them was a reason to refuse, and the advice -- "leave only the block being registered on the
    table" -- was impossible to follow, because the operator cannot remove something that was
    never there.
    """
    on_table = {18: (700.0, 400.0), 85: (120.0, 90.0), 182: (1500.0, 700.0), 190: (400.0, 750.0)}

    assert marker_on_target(on_table, (700.0, 400.0)) == 18


def test_another_building_already_on_the_table_does_not_interfere():
    """Registering G11 while G07 sits at the other end is an ordinary thing to want to do."""
    on_table = {18: (700.0, 400.0), 12: (760.0, 400.0)}

    assert marker_on_target(on_table, (700.0, 400.0)) == 18


def test_the_nearest_marker_wins_when_two_are_close():
    on_table = {18: (700.0, 400.0), 24: (760.0, 400.0)}

    assert marker_on_target(on_table, (740.0, 400.0)) == 24


def test_a_block_that_is_not_on_the_target_is_refused_with_the_distance():
    """Told in centimetres, because that is what the operator can act on at the table."""
    on_table = {18: (1200.0, 400.0)}

    with pytest.raises(ValueError, match="50.0 cm"):
        marker_on_target(on_table, (700.0, 400.0))


def test_an_empty_table_says_so_rather_than_reporting_a_distance():
    with pytest.raises(ValueError, match="no marker is on the table"):
        marker_on_target({}, (700.0, 400.0))


def test_the_proximity_window_covers_a_block_laid_down_by_hand():
    """A 1:500 block is 4-9 cm across; the marker's centre is not the block's centre."""
    on_table = {18: (700.0 + 90.0, 400.0)}

    assert marker_on_target(on_table, (700.0, 400.0)) == 18


# --- what is on the table *now* ----------------------------------------------------------
#
# The bug these pin down (2026-09-04 rig): registration compared the target against memories
# rather than against the table. `latest_marker_pixels` carried no timestamp and was never
# expired, so a marker seen ten times at any point since the process started stayed a candidate
# forever, at wherever it had last been seen. An operator with the block dead centre on the
# turquoise was told "the nearest (marker 182) is 37.4 cm away" and could do nothing about it,
# because 182 was not on the table to be moved.


def _sighting(x, y, age_seconds, now=1000.0):
    return (x, y, now - age_seconds)


def _timed_samples(count, age_seconds, rotation=-116.25, now=1000.0):
    return [(now - age_seconds, rotation) for _ in range(count)]


def test_a_marker_last_seen_minutes_ago_is_not_on_the_table():
    """The ghost that made the 2026-09-04 rig session unregisterable."""
    on_table = markers_on_the_table(
        {182: _sighting(500.0, 400.0, age_seconds=300.0)},
        {182: _timed_samples(30, age_seconds=300.0)},
        now=1000.0,
    )

    assert on_table == {}


def test_a_ghost_sitting_exactly_on_the_target_does_not_get_registered_instead():
    """The quiet half of the bug, and the worse half.

    A block that used to sit on the target leaves its last pixel behind forever. The operator
    puts the *right* block down a few centimetres off, and the memory -- being dead centre --
    wins the proximity contest. Nothing refuses, nothing warns: the catalog gets marker 182's
    heading filed as G11's true-north reference, which is precisely the silent constant error
    this whole flow exists to end.
    """
    on_table = markers_on_the_table(
        {
            18: _sighting(870.0, 400.0, age_seconds=0.4),
            182: _sighting(820.7, 399.9, age_seconds=300.0),
        },
        {
            18: _timed_samples(12, age_seconds=0.4),
            182: _timed_samples(30, age_seconds=300.0),
        },
        now=1000.0,
    )

    assert marker_on_target(on_table, (820.7, 399.9)) == 18


def test_a_marker_seen_now_but_only_a_few_times_is_not_steady_enough():
    """Under the sample floor there is no averaging left to beat corner noise."""
    on_table = markers_on_the_table(
        {18: _sighting(820.0, 400.0, age_seconds=0.2)},
        {18: _timed_samples(MINIMUM_REFERENCE_SAMPLES - 1, age_seconds=0.2)},
        now=1000.0,
    )

    assert on_table == {}


def test_readings_that_have_aged_out_do_not_count_towards_the_sample_floor():
    """Ten readings from five minutes ago are not two seconds of the block sitting still."""
    stale = _timed_samples(MINIMUM_REFERENCE_SAMPLES, age_seconds=300.0)
    fresh = _timed_samples(2, age_seconds=0.2)

    on_table = markers_on_the_table(
        {18: _sighting(820.0, 400.0, age_seconds=0.2)},
        {18: stale + fresh},
        now=1000.0,
    )

    assert on_table == {}


def test_calibration_markers_are_never_candidates():
    on_table = markers_on_the_table(
        {200: _sighting(820.0, 400.0, age_seconds=0.2)},
        {200: _timed_samples(30, age_seconds=0.2)},
        now=1000.0,
        reserved_ids={200},
    )

    assert on_table == {}


def test_the_refusal_names_every_marker_it_could_see_and_how_far_off_it_was():
    """A refusal the operator cannot act on is the failure, not the diagnosis.

    "the nearest (marker 182) is 37.4 cm away" is unactionable twice over: it does not say what
    else was on the table, and it does not say the block the operator is holding was not seen at
    all -- which is the thing they would have to fix.
    """
    on_table = {182: (1200.0, 400.0), 24: (700.0, 900.0)}

    with pytest.raises(ValueError) as refusal:
        marker_on_target(on_table, (700.0, 400.0))

    assert "182" in str(refusal.value) and "50.0 cm" in str(refusal.value)
    assert "24" in str(refusal.value)


def test_an_empty_table_says_nothing_was_seen_rather_than_reporting_a_distance():
    with pytest.raises(ValueError, match="no marker is on the table"):
        marker_on_target({}, (700.0, 400.0))


# --- the operator naming the marker outright --------------------------------------------
#
# Proximity depends on a chain nothing can check from inside the code: AOI centre -> projector
# -> physical table -> camera -> stitched pixel. When any link is off, the refusal is the same
# and there is nothing to do about it. Naming the id closes the loop deterministically.


def test_a_named_marker_that_is_on_the_table_is_simply_used():
    assert named_marker(empty_catalog(), "G11", 18, {18: (820.0, 400.0)}) == 18


def test_naming_the_marker_a_building_already_owns_is_how_it_gets_re_registered():
    """A re-glued or mis-registered block: the operator fixes it by registering it again."""
    catalog = _catalog_with(catalog_entry(_feature(), [18], {18: -116.25}))

    assert named_marker(catalog, "G11", 18, {18: (820.0, 400.0)}) == 18


def test_naming_a_marker_another_building_owns_is_refused_by_name():
    catalog = _catalog_with(catalog_entry(_feature("G07", "B-07"), [12], {12: 5.0}))

    with pytest.raises(ValueError, match="G07"):
        named_marker(catalog, "G11", 12, {12: (820.0, 400.0)})


def test_naming_a_marker_that_is_not_on_the_table_is_refused():
    """Otherwise the reference would be averaged out of readings of something long gone."""
    with pytest.raises(ValueError, match="not on the table"):
        named_marker(empty_catalog(), "G11", 18, {24: (820.0, 400.0)})
