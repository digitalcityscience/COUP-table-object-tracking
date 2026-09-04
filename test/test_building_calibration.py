"""Per-building calibration: the residual left over once the global mapping is right.

Step 1 closed the scale error, so what remains per building is the small stuff -- the marker not
glued exactly at the block's bbox centre, the block's reference heading being a degree or two
off. Those are *constants of the building*, stored in the catalog in the catalog's own units
(real-world metres in the building's local frame), never in table pixels and never in world
axes: they have to turn with the block.
"""

from pathlib import Path
import math
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physical_building_catalog import (
    BUILDING_CALIBRATION_FIELDS,
    DEFAULT_BUILDING_CALIBRATION,
    alignment_is_verified,
    apply_building_calibration,
    building_calibration_of,
    calibration_as_message_fields,
    calibration_from_message,
    building_feature,
    catalog_entry,
    empty_catalog,
    geometry_bbox,
    load_catalog,
    place_geometry,
    save_catalog,
    table_millimetres_to_local_metres,
)


def _feature():
    ring = [
        [10.0, 53.0],
        [10.0002, 53.0],
        [10.0002, 53.0001],
        [10.0, 53.0001],
        [10.0, 53.0],
    ]
    return {
        "type": "Feature",
        "properties": {"building_id": "G17", "city_scope_id": "B-17"},
        "geometry": {"type": "Polygon", "coordinates": [ring]},
    }


def _entry(**calibration):
    entry = catalog_entry(_feature(), [24], {24: 0.0})
    if calibration:
        entry = apply_building_calibration(entry, calibration)
    return entry


def _local_centre(geometry) -> tuple[float, float]:
    min_x, min_y, max_x, max_y = geometry_bbox(geometry)
    return (min_x + max_x) / 2, (min_y + max_y) / 2


# --- the unit the panel nudges in -------------------------------------------------------


def test_one_table_millimetre_is_half_a_real_world_metre():
    """The panel's arrow key is 1 px = 1 mm on the table; the catalog stores real metres."""
    assert table_millimetres_to_local_metres(1.0) == pytest.approx(0.5)
    assert table_millimetres_to_local_metres(-10.0) == pytest.approx(-5.0)


def test_a_message_is_translated_from_the_operator_s_units_into_the_catalog_s():
    """The panel speaks table millimetres; the catalog speaks real metres. One conversion point."""
    translated = calibration_from_message(
        {
            "type": "building_calibration",
            "building_id": "G17",
            "marker_id": 24,
            "rotation_offset_deg": -2.5,
            "offset_east_mm": 0.7,
            "offset_north_mm": -0.24,
            "scale_residual": 1.01,
        }
    )

    assert translated == {
        "rotation_offset_deg": pytest.approx(-2.5),
        "offset_east_m": pytest.approx(0.35),
        "offset_north_m": pytest.approx(-0.12),
        "scale_residual": pytest.approx(1.01),
    }


def test_a_message_translates_only_the_fields_it_actually_carries():
    """A partial save must stay partial all the way through, or the merge is pointless."""
    assert calibration_from_message({"offset_north_mm": -0.4}) == {
        "offset_north_m": pytest.approx(-0.2)
    }
    assert calibration_from_message({"type": "building_calibration"}) == {}


def test_the_stored_calibration_round_trips_back_into_the_message_s_units():
    """The panel has to be able to open showing where a building actually stands.

    Without this the panel could only start from zero, and since Python *replaces* each field it
    receives, the operator's next save would wipe the previous sitting's measurements rather than
    refine them. So the inverse conversion is a correctness requirement, not a convenience.
    """
    message = {
        "rotation_offset_deg": -2.5,
        "offset_east_mm": 0.7,
        "offset_north_mm": -0.24,
        "scale_residual": 1.01,
    }
    entry = apply_building_calibration(_entry(), calibration_from_message(message))

    assert calibration_as_message_fields(entry) == {
        "rotation_offset_deg": pytest.approx(-2.5),
        "offset_east_mm": pytest.approx(0.7),
        "offset_north_mm": pytest.approx(-0.24),
        "scale_residual": pytest.approx(1.01),
    }


def test_an_uncalibrated_building_reports_a_neutral_message_calibration():
    """Neutral on the three fields that have a neutral value, and *empty* on the one that has not.

    `rotation_offset_deg` comes across as null rather than `0.0` so the panel opens its rotation
    control blank and labelled instead of showing a confident zero nobody measured.
    """
    assert calibration_as_message_fields(_entry()) == {
        "rotation_offset_deg": None,
        "offset_east_mm": 0.0,
        "offset_north_mm": 0.0,
        "scale_residual": 1.0,
    }


# --- catalog storage --------------------------------------------------------------------


def test_a_fresh_catalog_entry_has_a_neutral_calibration():
    """An uncalibrated building must draw exactly as it did before this step existed."""
    assert building_calibration_of(_entry()) == DEFAULT_BUILDING_CALIBRATION
    assert DEFAULT_BUILDING_CALIBRATION == {
        "rotation_offset_deg": None,
        "offset_east_m": 0.0,
        "offset_north_m": 0.0,
        "scale_residual": 1.0,
    }


def test_applying_a_calibration_stores_exactly_the_four_declared_fields():
    entry = _entry(rotation_offset_deg=-2.5, offset_east_m=0.35, offset_north_m=-0.12, scale_residual=1.01)

    stored = building_calibration_of(entry)
    assert set(stored) == set(BUILDING_CALIBRATION_FIELDS)
    assert stored["rotation_offset_deg"] == pytest.approx(-2.5)
    assert stored["offset_east_m"] == pytest.approx(0.35)
    assert stored["scale_residual"] == pytest.approx(1.01)


def test_applying_a_partial_calibration_leaves_the_other_fields_alone():
    """The panel saves whichever axis the operator touched; the rest must not silently reset."""
    entry = _entry(rotation_offset_deg=-2.5, offset_east_m=0.35)

    nudged = apply_building_calibration(entry, {"offset_north_m": -0.12})

    assert building_calibration_of(nudged)["rotation_offset_deg"] == pytest.approx(-2.5)
    assert building_calibration_of(nudged)["offset_east_m"] == pytest.approx(0.35)
    assert building_calibration_of(nudged)["offset_north_m"] == pytest.approx(-0.12)


def test_applying_a_calibration_does_not_mutate_the_entry_it_was_given():
    entry = _entry()

    apply_building_calibration(entry, {"rotation_offset_deg": -2.5})

    assert building_calibration_of(entry)["rotation_offset_deg"] is None


def test_an_unknown_calibration_field_is_refused():
    """A typo'd field would be stored, ignored at runtime, and never noticed."""
    with pytest.raises(ValueError, match="offset_up_m"):
        apply_building_calibration(_entry(), {"offset_up_m": 1.0})


def test_a_non_positive_scale_residual_is_refused():
    with pytest.raises(ValueError, match="scale_residual"):
        apply_building_calibration(_entry(), {"scale_residual": 0.0})


def test_a_calibrated_catalog_survives_a_save_and_reload(tmp_path):
    """Step 3's done-condition: a value saved from the panel is still there after a restart."""
    path = tmp_path / "physical-building-catalog.json"
    catalog = empty_catalog()
    catalog["buildings"].append(
        _entry(rotation_offset_deg=-2.5, offset_east_m=0.35, offset_north_m=-0.12, scale_residual=1.01)
    )

    save_catalog(path, catalog)
    reloaded = load_catalog(path)

    assert building_calibration_of(reloaded["buildings"][0]) == {
        "rotation_offset_deg": pytest.approx(-2.5),
        "offset_east_m": pytest.approx(0.35),
        "offset_north_m": pytest.approx(-0.12),
        "scale_residual": pytest.approx(1.01),
    }


# --- the runtime chain ------------------------------------------------------------------


def test_the_rotation_offset_lands_on_top_of_the_marker_reference():
    """`effective_rotation = detected - marker_reference + rotation_offset_deg`, in that order."""
    entry = catalog_entry(_feature(), [24], {24: 170.0})
    entry = apply_building_calibration(entry, {"rotation_offset_deg": -2.5})

    result = building_feature(entry, 24, (10.0, 53.0), -170.0)

    assert result["properties"]["rotation"] == pytest.approx(20.0 - 2.5)


def test_the_scale_residual_multiplies_the_session_s_global_factor():
    entry = apply_building_calibration(_entry(), {"scale_residual": 1.10})

    plain_width, _ = _extent(building_feature(_entry(), 24, (10.0, 53.0), 0.0, scale=0.5)["geometry"])
    residual_width, _ = _extent(building_feature(entry, 24, (10.0, 53.0), 0.0, scale=0.5)["geometry"])

    assert residual_width == pytest.approx(plain_width * 1.10, rel=1e-3)
    assert building_feature(entry, 24, (10.0, 53.0), 0.0, scale=0.5)["properties"][
        "model_scale_factor"
    ] == pytest.approx(0.55)


def _extent(geometry) -> tuple[float, float]:
    min_x, min_y, max_x, max_y = geometry_bbox(geometry)
    return max_x - min_x, max_y - min_y


def test_the_offset_moves_the_footprint_off_the_detected_marker_centre():
    """The marker is not glued exactly at the bbox centre; this is that gap, in local metres."""
    centre = (10.0, 53.0)
    upright = place_geometry(_entry()["geometry"], centre, 0.0)
    shifted = place_geometry(_entry()["geometry"], centre, 0.0, offset=(5.0, -3.0))

    # An uncalibrated footprint is centred on the detected marker position exactly.
    assert _displacement_metres(centre, _local_centre(upright)) == pytest.approx((0.0, 0.0), abs=0.02)
    east, north = _displacement_metres(centre, _local_centre(shifted))
    assert east == pytest.approx(5.0, abs=0.05)
    assert north == pytest.approx(-3.0, abs=0.05)


def test_the_offset_turns_with_the_block_rather_than_with_the_compass():
    """The decision this locks: 'kaydırma yerel çerçevede saklanır (blokla birlikte döner)'.

    A 5 m east offset on an upright block must become a 5 m *north* offset once the block is
    turned 90 degrees. If it were stored in world axes it would stay pointing east, and every
    per-building constant would be wrong the moment the operator turned the block.
    """
    centre = (10.0, 53.0)
    entry = _entry()

    upright = place_geometry(entry["geometry"], centre, 0.0, offset=(5.0, 0.0))
    turned = place_geometry(entry["geometry"], centre, 90.0, offset=(5.0, 0.0))
    turned_reference = place_geometry(entry["geometry"], centre, 90.0)

    upright_displacement = _displacement_metres(centre, _local_centre(upright))
    turned_displacement = _displacement_metres(centre, _local_centre(turned))
    reference_displacement = _displacement_metres(centre, _local_centre(turned_reference))

    # Upright: the offset points east.
    assert upright_displacement[0] == pytest.approx(5.0, abs=0.05)
    assert upright_displacement[1] == pytest.approx(0.0, abs=0.05)
    # Turned 90 degrees: the same stored offset now points north, relative to where the
    # uncalibrated turned footprint already sits.
    delta_east = turned_displacement[0] - reference_displacement[0]
    delta_north = turned_displacement[1] - reference_displacement[1]
    assert math.hypot(delta_east, delta_north) == pytest.approx(5.0, abs=0.05)
    assert delta_east == pytest.approx(0.0, abs=0.05)
    assert delta_north == pytest.approx(5.0, abs=0.05)


def _displacement_metres(origin, target) -> tuple[float, float]:
    """East/north metres from `origin` to `target`, both `(lng, lat)`."""
    from pyproj import Geod

    azimuth, _back, distance = Geod(ellps="WGS84").inv(origin[0], origin[1], target[0], target[1])
    radians = math.radians(azimuth)
    return distance * math.sin(radians), distance * math.cos(radians)


def test_the_offset_is_scaled_onto_the_table_along_with_the_footprint():
    """A local-metre offset is catalog-sized, so the drawing shrinks it exactly like the shape."""
    entry = _entry()
    centre = (10.0, 53.0)

    full = place_geometry(entry["geometry"], centre, 0.0, offset=(5.0, 0.0))
    half = place_geometry(entry["geometry"], centre, 0.0, offset=(5.0, 0.0), scale=0.5)

    full_east, _ = _displacement_metres(centre, _local_centre(full))
    half_east, _ = _displacement_metres(centre, _local_centre(half))
    assert half_east == pytest.approx(full_east * 0.5, rel=1e-3)


def test_an_uncalibrated_building_is_drawn_exactly_as_before():
    """The neutral calibration must be a true no-op, not an approximate one."""
    entry = _entry()
    plain = building_feature(entry, 24, (10.0, 53.0), 12.5, scale=0.654)
    explicit = building_feature(
        apply_building_calibration(entry, DEFAULT_BUILDING_CALIBRATION),
        24,
        (10.0, 53.0),
        12.5,
        scale=0.654,
    )
    assert plain["geometry"] == explicit["geometry"]


def test_the_real_catalog_still_loads_without_any_calibration_block():
    """The three registered buildings predate this step; they must keep working untouched."""
    catalog = load_catalog(Path(__file__).resolve().parents[1] / "physical-building-catalog.json")

    for building in catalog["buildings"]:
        assert building_calibration_of(building) == DEFAULT_BUILDING_CALIBRATION


# --- D1: a measured zero and an unmeasured heading are not the same thing ----------------


def test_a_freshly_registered_building_reports_its_alignment_as_unverified():
    """The D1 bug: registration declares the block's random resting heading to be true north.

    The maths downstream is exact, which is what made this invisible -- every registration looked
    perfect and every building could still be turned by a constant of up to 180 degrees. Nothing
    can fix that automatically; the only thing code can do is refuse to keep quiet about it.
    """
    assert alignment_is_verified(_entry()) is False
    assert building_feature(_entry(), 24, (10.0, 53.0), 0.0)["properties"][
        "alignment_verified"
    ] is False


def test_measuring_the_offset_is_what_marks_a_building_aligned():
    assert alignment_is_verified(_entry(rotation_offset_deg=-2.5)) is True
    assert building_feature(_entry(rotation_offset_deg=-2.5), 24, (10.0, 53.0), 0.0)[
        "properties"
    ]["alignment_verified"] is True


def test_a_measured_zero_is_verified_and_an_unmeasured_one_is_not():
    """The whole point of the None: `0.0` used to mean both of these at once."""
    assert alignment_is_verified(_entry(rotation_offset_deg=0.0)) is True
    assert alignment_is_verified(_entry()) is False


def test_nudging_another_axis_does_not_silently_claim_the_heading_was_measured():
    """An operator saving an east offset has not looked at which way the building faces."""
    nudged = _entry(offset_east_m=0.35)

    assert alignment_is_verified(nudged) is False
    assert building_calibration_of(nudged)["offset_east_m"] == pytest.approx(0.35)


def test_an_unverified_building_still_draws_and_applies_no_rotation_offset():
    """Refusing to draw would be worse than drawing a marked, suspect footprint."""
    unverified = building_feature(_entry(), 24, (10.0, 53.0), 30.0)
    zeroed = building_feature(_entry(rotation_offset_deg=0.0), 24, (10.0, 53.0), 30.0)

    assert unverified["properties"]["rotation"] == pytest.approx(30.0)
    assert unverified["geometry"] == zeroed["geometry"]


def test_a_building_can_be_put_back_to_unmeasured():
    """The panel needs a way to retract a claim, and `float(None)` would record a measured zero."""
    entry = _entry(rotation_offset_deg=-2.5)

    retracted = apply_building_calibration(
        entry, calibration_from_message({"rotation_offset_deg": None})
    )

    assert alignment_is_verified(retracted) is False


def test_the_unmeasured_heading_survives_a_save_and_reload(tmp_path):
    """A stored `null` and a missing key must read back the same, or a restart re-hides the gap."""
    path = tmp_path / "physical-building-catalog.json"
    catalog = empty_catalog()
    catalog["buildings"].append(_entry(offset_east_m=0.35))

    save_catalog(path, catalog)
    reloaded = load_catalog(path)

    assert alignment_is_verified(reloaded["buildings"][0]) is False


def test_the_three_registered_buildings_are_all_still_unaligned():
    """Documents the actual state of the rig: no absolute heading has ever been verified."""
    catalog = load_catalog(Path(__file__).resolve().parents[1] / "physical-building-catalog.json")

    assert [building["building_id"] for building in catalog["buildings"] if not alignment_is_verified(building)] == [
        building["building_id"] for building in catalog["buildings"]
    ]


# --- D2: the heading goes through the homography, not around it -------------------------


def test_the_detected_angle_and_its_reference_go_through_the_same_conversion():
    """Converting only the detected angle would subtract two numbers from different frames.

    `build.py` records `marker_reference_rotations` in raw table-pixel degrees -- it has no
    homography to consult. So a conversion applied to the detected heading alone injects the
    table's own tilt as a fresh systematic error, and forces every registered reference to be
    migrated. Running both through the same callable makes the difference the real local
    world-frame angle and leaves the stored references valid as they stand.
    """
    entry = catalog_entry(_feature(), [24], {24: 10.0})
    seen = []

    def to_map(table_degrees):
        seen.append(table_degrees)
        return table_degrees + 3.0  # a pure rotation: it must cancel out of the difference

    result = building_feature(entry, 24, (10.0, 53.0), 40.0, table_direction_to_map=to_map)

    assert seen == [40.0, 10.0]
    assert result["properties"]["rotation"] == pytest.approx(30.0)


def test_a_non_rotational_conversion_reaches_the_drawn_rotation():
    """The 1.5 degrees D2 buys: shear and anisotropy do *not* cancel out of the difference."""
    entry = catalog_entry(_feature(), [24], {24: 0.0})

    result = building_feature(
        entry, 24, (10.0, 53.0), 90.0, table_direction_to_map=lambda deg: deg * 1.02
    )

    assert result["properties"]["rotation"] == pytest.approx(91.8)


def test_omitting_the_conversion_leaves_the_scalar_path_exactly_as_it_was():
    """The seam has to be free for every caller that has no homography, including the tests."""
    entry = catalog_entry(_feature(), [24], {24: 10.0})

    assert (
        building_feature(entry, 24, (10.0, 53.0), 40.0)["geometry"]
        == building_feature(
            entry, 24, (10.0, 53.0), 40.0, table_direction_to_map=lambda deg: deg
        )["geometry"]
    )
