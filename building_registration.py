"""Registering a physical block against the building it stands for.

Registration answers one question and it is not the one it looks like. Measuring the block's
heading is trivial -- a camera does it thirty times a second. What registration has to establish
is the *relationship* between that heading and the direction the real building faces, and no
amount of looking at an ArUco marker can produce it: the marker could be glued to the block
straight, turned 90 degrees, or upside down, and the four cases are indistinguishable from the
angle alone. The information is not in the data. A human has to supply it exactly once.

Until 2026-09-04 nobody did, and nothing said so. `build.py` recorded whatever heading the block
happened to be lying at and the system took that pose to be the building's true-north orientation
-- a constant error of up to 180 degrees, invisible because every registration then looked
arithmetically perfect (`detected == reference` draws on the catalog's real heading to 0.000
degrees, which is the tell, not the reassurance).

The fix is to stop asking the operator for a number and start giving them a picture. The frontend
projects the building's real footprint onto the table at its real heading; the operator turns the
block until it is *parallel* to that projection and confirms. At that instant the block's measured
heading genuinely is the catalog's true-north heading, so the reference is correct by
construction and `rotation_offset_deg` is a measured `0.0` rather than an unmeasured `None`.

Parallel, not on top of: the reference needs only the angle -- position comes from live tracking
every frame -- so the block may sit anywhere on the table while being aligned. That is what lets
an operator register at the left, centre and right of the table in turn and see whether both
cameras read the same heading.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

from physical_building_catalog import (
    apply_building_calibration,
    catalog_entry,
    marker_index,
)

#: How far apart two readings of the same stationary marker may be before the sample set is
#: refused, in degrees.
#:
#: A block being registered is not moving, so the only spread in its heading is corner-detection
#: noise -- measured at +/-1.3 degrees on a good run and +/-6 on a bad one (2026-09-04). Well
#: above that, and the marker was still being moved when the operator confirmed, or two readings
#: of different physical objects have been merged. Either way the mean is not a heading and
#: filing it would bake the error into the catalog as a constant, which is the exact failure this
#: module exists to end.
MAXIMUM_REFERENCE_SPREAD_DEG = 20.0

#: How few readings still make a reference. Snapshots arrive every 200 ms, so ten is two seconds
#: of the block sitting still -- enough for the circular mean to beat single-frame corner noise,
#: short enough that the operator is not left holding a block.
MINIMUM_REFERENCE_SAMPLES = 10


def circular_mean_degrees(angles: Iterable[float]) -> float:
    """The mean of angles that wrap, in degrees.

    A plain arithmetic mean is wrong here and wrong in a way that hides: two readings of the same
    stationary marker at 179 and -179 degrees are two degrees apart and average to 0, pointing the
    block backwards. Averaging the unit vectors instead has no seam.
    """
    values = list(angles)
    if not values:
        raise ValueError("cannot average an empty set of angles")
    sine = sum(math.sin(math.radians(angle)) for angle in values)
    cosine = sum(math.cos(math.radians(angle)) for angle in values)
    return math.degrees(math.atan2(sine, cosine))


def angular_spread_degrees(angles: Iterable[float]) -> float:
    """How far the widest reading sits from the circular mean, in degrees.

    Measured against the mean rather than as max-minus-min so that the wrap seam cannot inflate
    it: 179 and -179 are two degrees apart, and this says so.
    """
    values = list(angles)
    mean = circular_mean_degrees(values)
    return max(abs((angle - mean + 180.0) % 360.0 - 180.0) for angle in values)


def reference_rotation_from_samples(samples: Iterable[float]) -> float:
    """The heading to file as a building's reference, or raise explaining why these readings are not one."""
    values = list(samples)
    if len(values) < MINIMUM_REFERENCE_SAMPLES:
        raise ValueError(
            f"only {len(values)} reading(s) of this marker; {MINIMUM_REFERENCE_SAMPLES} are needed "
            "to average out corner-detection noise. Keep the block still and in view, then confirm again"
        )
    spread = angular_spread_degrees(values)
    if spread > MAXIMUM_REFERENCE_SPREAD_DEG:
        raise ValueError(
            f"this marker's heading moved {spread:.1f} deg across the readings, more than the "
            f"{MAXIMUM_REFERENCE_SPREAD_DEG:.0f} deg a still block can wander. The block was still "
            "being moved, or two objects were merged; settle it and confirm again"
        )
    return circular_mean_degrees(values)


#: How far from the alignment target a marker may sit and still be the one being registered, in
#: table pixels (one pixel is one millimetre).
#:
#: A 1:500 block is 4-9 cm across, so 12 cm comfortably covers a block laid on the target by hand
#: -- including the gap between the marker's centre and the block's own centre -- while excluding
#: anything sitting somewhere else on the table.
TARGET_PROXIMITY_PX = 120.0


def marker_on_target(
    marker_pixels: Mapping[int, tuple[float, float]],
    target_pixel: tuple[float, float],
    proximity_px: float = TARGET_PROXIMITY_PX,
) -> int:
    """Which marker is sitting on the alignment target.

    This is the question registration actually has to answer, and the operator has already
    answered it physically by putting the block on the projected footprint. Reading it off the
    position rather than by elimination is what makes everything else on the table irrelevant:
    other blocks, and the spurious ArUco reads a noisy frame produces, are simply not on the
    target. Elimination made every one of them a reason to refuse -- including phantoms the
    operator cannot remove, because they were never there.
    """
    target_x, target_y = float(target_pixel[0]), float(target_pixel[1])
    distances = {
        marker_id: math.hypot(float(x) - target_x, float(y) - target_y)
        for marker_id, (x, y) in marker_pixels.items()
    }
    near = {marker_id: d for marker_id, d in distances.items() if d <= proximity_px}
    if not near:
        closest = min(distances.items(), key=lambda item: item[1], default=None)
        if closest is None:
            raise ValueError(
                "no marker is on the table at all. Put the block on the projected target and "
                "confirm again"
            )
        raise ValueError(
            f"no marker is on the target; the nearest (marker {closest[0]}) is "
            f"{closest[1] / 10:.1f} cm away. Put the block on the projected outline and confirm again"
        )
    return min(near.items(), key=lambda item: item[1])[0]


def marker_to_register(
    catalog: Mapping[str, Any],
    building_id: str,
    seen_marker_ids: Iterable[int],
    reserved_marker_ids: Iterable[int],
) -> int:
    """Which marker on the table is the one being registered.

    Resolved here rather than asked of the frontend because the frontend cannot know: a building
    being registered for the first time has no catalog entry, so there is no marker id to send.
    What it *is*, by elimination, is the one marker on the table that no building already claims
    -- other blocks may legitimately be sitting there and each of those is already spoken for.

    Re-registering an existing building is the exception: its marker is already catalogued, so it
    is claimed by this same building and that is what identifies it.
    """
    reserved = set(reserved_marker_ids)
    owners = marker_index(dict(catalog))
    candidates = [
        marker_id
        for marker_id in sorted(set(seen_marker_ids))
        if marker_id not in reserved
        and (marker_id not in owners or owners[marker_id]["building_id"] == building_id)
    ]
    if not candidates:
        raise ValueError(
            f"no unclaimed marker is on the table. Put {building_id}'s block on the table, or "
            "take away the blocks whose buildings are already registered"
        )
    if len(candidates) > 1:
        raise ValueError(
            f"markers {candidates} are all unclaimed, so there is no telling which one is "
            f"{building_id}. Leave only the block being registered on the table"
        )
    return candidates[0]


def registered_entry(
    source_feature: Mapping[str, Any],
    marker_id: int,
    reference_rotation: float,
) -> dict[str, Any]:
    """One catalog entry for a block that was confirmed aligned to its own projected footprint.

    `rotation_offset_deg` is set to a measured `0.0`, not left unmeasured, and that is the whole
    point of the flow: the operator has just been shown the building's real heading and has put
    the block on it, so the residual between the reference and the truth is nil *and somebody
    checked*. Writing `None` here would be as wrong as the old code writing `0.0` blind -- it
    would throw away the one measurement the procedure exists to take.
    """
    entry = catalog_entry(dict(source_feature), [marker_id], {marker_id: float(reference_rotation)})
    return apply_building_calibration(entry, {"rotation_offset_deg": 0.0})


def catalog_with_entry(catalog: Mapping[str, Any], entry: Mapping[str, Any]) -> dict[str, Any]:
    """`catalog` with `entry` in place of whatever it held for that building.

    Replaces rather than appends, so re-registering a block that was re-glued does not leave the
    old marker still claiming the building. Returns a new dict: a refused write must never have
    half-mutated the live catalog.
    """
    building_id = entry["building_id"]
    updated = dict(catalog)
    updated["buildings"] = [
        building for building in catalog["buildings"] if building["building_id"] != building_id
    ] + [dict(entry)]
    return updated
