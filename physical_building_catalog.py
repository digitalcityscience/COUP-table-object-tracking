"""Physical-building catalog normalization and runtime placement helpers."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Iterable

from pyproj import Geod


CATALOG_VERSION = 2
COORDINATE_SYSTEM = {
    "type": "local_cartesian",
    "unit": "meter",
    "anchor": "bbox_center",
    "axis": ["east", "north"],
}
_GEOD = Geod(ellps="WGS84")

#: The four per-building calibration fields, in the order the panel presents them. The single
#: declared list: `apply_building_calibration` validates against it, so a typo'd field is refused
#: at the boundary instead of being stored, ignored at runtime, and never noticed.
BUILDING_CALIBRATION_FIELDS = (
    "rotation_offset_deg",
    "offset_east_m",
    "offset_north_m",
    "scale_residual",
)

#: What an unmeasured absolute alignment is, as distinct from a measured zero. `0.0` used to sit
#: here and it was the D1 bug: "we measured, the offset really is nil" and "nobody has ever
#: checked which way this block faces" collapsed onto the same number, so a building whose
#: `marker_reference_rotations` were captured at whatever random heading the block happened to be
#: lying at during registration drew with a constant error of up to 180 degrees and nothing --
#: not the catalog, not the feature, not the panel -- said so.
#:
#: `None` is a load-bearing value, not a missing one. `building_feature` still draws an
#: unmeasured building (refusing to draw would be worse), but it applies no offset and stamps
#: `alignment_verified: false` on the feature so the projection and the panel can mark it.
UNMEASURED_ROTATION_OFFSET = None

#: What an uncalibrated building means. A true no-op, not an approximate one: the three buildings
#: registered before this step existed carry no calibration block at all and must keep drawing
#: byte-identically.
DEFAULT_BUILDING_CALIBRATION = {
    "rotation_offset_deg": UNMEASURED_ROTATION_OFFSET,
    "offset_east_m": 0.0,
    "offset_north_m": 0.0,
    "scale_residual": 1.0,
}

#: The calibration fields that carry `None` for "not measured yet" rather than a neutral number.
#: Only the rotation needs it: an unmeasured east/north offset or scale residual genuinely is
#: neutral (the block is where the marker says it is, at the size the session's scale says), while
#: an unmeasured *heading* is a claim about the real world that nobody has checked.
NULLABLE_BUILDING_CALIBRATION_FIELDS = ("rotation_offset_deg",)

#: The scale the physical blocks on the table are milled at: 1:500. The single declared value --
#: the catalog stores real-world metres and the table map is drawn at whatever ground scale the
#: operator's AOI happens to land on, so this is the only place the two are reconciled. Verified
#: to 1-2 mm on two blocks (G07, G17) with a ruler; G11 is still unverified.
MODEL_SCALE = 500


def _positions(geometry: dict[str, Any]) -> Iterable[list[float]]:
    geometry_type = geometry.get("type")
    coordinate_key = "local_coordinates" if "local_coordinates" in geometry else "coordinates"
    coordinates = geometry.get(coordinate_key)
    if geometry_type == "Polygon":
        for ring in coordinates or []:
            yield from ring
    elif geometry_type == "MultiPolygon":
        for polygon in coordinates or []:
            for ring in polygon:
                yield from ring
    else:
        raise ValueError("Building geometry must be Polygon or MultiPolygon")


def geometry_bbox(geometry: dict[str, Any]) -> list[float]:
    positions = list(_positions(geometry))
    if not positions:
        raise ValueError("Building geometry has no coordinates")
    xs = [float(position[0]) for position in positions]
    ys = [float(position[1]) for position in positions]
    return [min(xs), min(ys), max(xs), max(ys)]


def _map_coordinates(
    geometry: dict[str, Any],
    convert,
    *,
    source_key: str,
    target_key: str,
) -> dict[str, Any]:
    result = {"type": geometry["type"]}
    coordinates = geometry[source_key]
    if result["type"] == "Polygon":
        result[target_key] = [[convert(position) for position in ring] for ring in coordinates]
    elif result["type"] == "MultiPolygon":
        result[target_key] = [
            [[convert(position) for position in ring] for ring in polygon]
            for polygon in coordinates
        ]
    else:
        raise ValueError("Building geometry must be Polygon or MultiPolygon")
    return result


def normalize_geometry(geometry: dict[str, Any]) -> tuple[dict[str, Any], list[float]]:
    """Convert WGS84 coordinates to local east/north metres around the WGS84 bbox centre."""
    min_lng, min_lat, max_lng, max_lat = geometry_bbox(geometry)
    anchor_lng = (min_lng + max_lng) / 2
    anchor_lat = (min_lat + max_lat) / 2

    def to_local(position: list[float]) -> list[float]:
        lng, lat = float(position[0]), float(position[1])
        azimuth, _back_azimuth, distance = _GEOD.inv(anchor_lng, anchor_lat, lng, lat)
        radians = math.radians(azimuth)
        return [distance * math.sin(radians), distance * math.cos(radians)]

    local_geometry = _map_coordinates(
        geometry,
        to_local,
        source_key="coordinates",
        target_key="local_coordinates",
    )
    local_bbox = geometry_bbox(local_geometry)
    return local_geometry, local_bbox


def catalog_entry(
    feature: dict[str, Any],
    marker_ids: list[int],
    marker_reference_rotations: dict[int, float] | None = None,
) -> dict[str, Any]:
    properties = copy.deepcopy(feature.get("properties", {}))
    building_id = str(properties.get("building_id", "")).strip().upper()
    if not building_id:
        raise ValueError("Source feature has no building_id")
    city_scope_id = properties.get("city_scope_id")
    if not isinstance(city_scope_id, str) or not city_scope_id:
        raise ValueError(f"Building {building_id} has no city_scope_id")
    geometry, local_bbox = normalize_geometry(feature["geometry"])
    reference_rotations = marker_reference_rotations or {}
    return {
        "building_id": building_id,
        "city_scope_id": city_scope_id,
        "marker_ids": sorted(set(marker_ids)),
        "marker_reference_rotations": {
            str(marker_id): float(reference_rotations.get(marker_id, 0.0))
            for marker_id in sorted(set(marker_ids))
        },
        "local_bbox": local_bbox,
        "geometry": geometry,
        "source_properties": properties,
    }


def empty_catalog() -> dict[str, Any]:
    return {"version": CATALOG_VERSION, "coordinate_system": copy.deepcopy(COORDINATE_SYSTEM), "buildings": []}


def load_catalog(path: Path) -> dict[str, Any]:
    if not path.exists():
        return empty_catalog()
    raw_catalog = path.read_text(encoding="utf-8")
    if not raw_catalog.strip():
        return empty_catalog()
    catalog = json.loads(raw_catalog)
    if catalog.get("version") != CATALOG_VERSION or catalog.get("coordinate_system") != COORDINATE_SYSTEM:
        raise ValueError(f"Unsupported physical-building catalog schema in {path}")
    if not isinstance(catalog.get("buildings"), list):
        raise ValueError(f"Catalog {path} has no buildings array")
    for building in catalog["buildings"]:
        geometry = building.get("geometry", {})
        if "local_coordinates" not in geometry or "coordinates" in geometry:
            raise ValueError(
                f"Building {building.get('building_id', '<unknown>')} must use geometry.local_coordinates"
            )
    return catalog


def save_catalog(path: Path, catalog: dict[str, Any]) -> None:
    catalog["buildings"] = sorted(catalog["buildings"], key=lambda item: item["building_id"])
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as output_file:
        json.dump(catalog, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")
    temporary_path.replace(path)


def marker_index(catalog: dict[str, Any]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for building in catalog["buildings"]:
        for marker_id in building["marker_ids"]:
            marker_id = int(marker_id)
            if marker_id in result:
                raise ValueError(
                    f"Marker {marker_id} is assigned to both {result[marker_id]['building_id']} and {building['building_id']}"
                )
            result[marker_id] = building
    return result


def table_millimetres_to_local_metres(millimetres: float) -> float:
    """One table millimetre, in the catalog's real-world local metres.

    The panel's fine adjustment is one arrow-key tick = 1 table pixel = 1 mm (10 px/cm), and the
    catalog stores real metres, so the two are exactly `MODEL_SCALE` apart. Named here rather
    than written out at the call site because it is the one place the panel's unit and the
    catalog's unit meet, and mixing them up is a factor-of-500 error that looks plausible on
    screen.
    """
    return millimetres / 1000 * MODEL_SCALE


#: The `building_calibration` message's fields, and the catalog field each one lands in. The
#: message speaks the operator's units -- the admin panel nudges in table millimetres, because
#: that is what an arrow key does and what a ruler on the table reads -- while the catalog stores
#: real-world metres. Keeping the translation here, rather than on the frontend, is what keeps
#: `MODEL_SCALE` a Python-only number: the frontend never needs to know the blocks are 1:500.
BUILDING_CALIBRATION_MESSAGE_FIELDS: dict[str, str] = {
    "rotation_offset_deg": "rotation_offset_deg",
    "offset_east_mm": "offset_east_m",
    "offset_north_mm": "offset_north_m",
    "scale_residual": "scale_residual",
}


def calibration_from_message(payload: dict[str, Any]) -> dict[str, float | None]:
    """The catalog-unit calibration a `building_calibration` message asks for.

    Only the fields the message actually carries, so a save of the one axis the operator touched
    stays a partial update (see `apply_building_calibration`). Millimetre fields are converted
    to catalog metres on the way through; degrees and the scale ratio pass straight along.

    An explicit `null` in a nullable field is carried through as `None` rather than coerced to
    zero: the panel needs a way to say "I was wrong to claim this was aligned" and put a building
    back to unmeasured, and `float(None)` would instead record a measured zero -- exactly the
    conflation this whole distinction exists to end.
    """
    calibration: dict[str, Any] = {}
    for message_field, catalog_field in BUILDING_CALIBRATION_MESSAGE_FIELDS.items():
        if message_field not in payload:
            continue
        raw_value = payload[message_field]
        if raw_value is None and catalog_field in NULLABLE_BUILDING_CALIBRATION_FIELDS:
            calibration[catalog_field] = None
            continue
        value = float(raw_value)
        calibration[catalog_field] = (
            table_millimetres_to_local_metres(value) if message_field.endswith("_mm") else value
        )
    return calibration


def calibration_as_message_fields(building: dict[str, Any]) -> dict[str, float | None]:
    """One building's stored calibration, back in the units a `building_calibration` speaks.

    The exact inverse of `calibration_from_message`, and published on every runtime feature so the
    admin panel can open showing where a building actually stands rather than at zero. Without it
    the panel has no way to read what Python already holds, and a second save built from a neutral
    draft would silently replace the first sitting's measurements instead of refining them.

    Named field-for-field after the message, so the value the panel receives is the value it sends
    back with no conversion of its own -- which is what keeps `MODEL_SCALE` a Python-only number.

    `rotation_offset_deg` comes across as JSON `null` when the building's heading has never been
    verified, so the panel can open its rotation control empty and labelled rather than showing a
    confident `0.0` that nobody ever measured.
    """
    stored = building_calibration_of(building)
    return {
        message_field: (
            local_metres_to_table_millimetres(stored[catalog_field])
            if message_field.endswith("_mm") and stored[catalog_field] is not None
            else stored[catalog_field]
        )
        for message_field, catalog_field in BUILDING_CALIBRATION_MESSAGE_FIELDS.items()
    }


def local_metres_to_table_millimetres(metres: float) -> float:
    """The inverse of `table_millimetres_to_local_metres` — catalog metres back to table mm."""
    return metres * 1000 / MODEL_SCALE


def building_calibration_of(building: dict[str, Any]) -> dict[str, Any]:
    """One building's calibration, defaulted field by field.

    Defaulted per field rather than per block, so a catalog entry that has only ever had its
    rotation nudged still reports a neutral offset and scale instead of nothing.

    `rotation_offset_deg` comes back as `None` when it has never been measured (see
    `UNMEASURED_ROTATION_OFFSET`); every other field is a float. A stored `null` and a missing
    key mean the same thing, which is what lets an old catalog written before this distinction
    existed read back as "unmeasured" rather than as "measured, and it was zero".
    """
    stored = building.get("calibration") or {}
    result: dict[str, Any] = {}
    for field in BUILDING_CALIBRATION_FIELDS:
        value = stored.get(field, DEFAULT_BUILDING_CALIBRATION[field])
        if value is None and field in NULLABLE_BUILDING_CALIBRATION_FIELDS:
            result[field] = None
        else:
            result[field] = float(value)
    return result


def alignment_is_verified(building: dict[str, Any]) -> bool:
    """Has anyone actually checked which way this building's block faces?

    The single reading of the `None`/number distinction `UNMEASURED_ROTATION_OFFSET` introduces,
    so "unmeasured" is asked for by name everywhere instead of being re-derived from a null check
    that is one refactor away from silently becoming `not offset` and treating a measured zero as
    unmeasured again.
    """
    return building_calibration_of(building)["rotation_offset_deg"] is not None


def applied_rotation_offset(calibration: dict[str, Any]) -> float:
    """The degrees a calibration actually contributes to a drawing: 0 when unmeasured.

    An unmeasured building still draws -- it is far more useful on the table with a known-suspect
    heading than absent -- so the arithmetic needs a number. This is the only place `None` becomes
    `0.0`, and it is deliberately not the same function that answers "is it verified".
    """
    offset = calibration["rotation_offset_deg"]
    return 0.0 if offset is None else float(offset)


def apply_building_calibration(
    building: dict[str, Any], calibration: dict[str, Any]
) -> dict[str, Any]:
    """A copy of `building` with `calibration`'s fields merged into its calibration block.

    A merge, not a replacement: the panel saves whichever axis the operator actually touched,
    and silently resetting the others would undo earlier measurements. Returns a copy so a
    rejected write can never have half-mutated the live in-memory catalog.
    """
    unknown = sorted(set(calibration) - set(BUILDING_CALIBRATION_FIELDS))
    if unknown:
        raise ValueError(
            f"Unknown building calibration field(s) {', '.join(unknown)}; "
            f"expected some of {', '.join(BUILDING_CALIBRATION_FIELDS)}"
        )
    merged = building_calibration_of(building)
    for field, value in calibration.items():
        if value is None and field in NULLABLE_BUILDING_CALIBRATION_FIELDS:
            merged[field] = None
        else:
            merged[field] = float(value)
    if not merged["scale_residual"] > 0:
        raise ValueError(
            f"scale_residual must be positive, got {merged['scale_residual']!r}"
        )
    updated = copy.deepcopy(building)
    updated["calibration"] = merged
    return updated


def model_scale_factor(ground_scale: float) -> float:
    """How far catalog geometry must shrink to land on top of its 1:500 block.

    The catalog is in real-world metres; the table map is drawn at `ground_scale` real-world
    centimetres per table centimetre (see `pixel_to_utm.ground_scale`), so unscaled catalog
    metres draw a building at `real_size / ground_scale` on the table while its block is
    `real_size / 500`. Multiplying the local coordinates by this factor makes the two equal:
    `(ground_scale / 500) / ground_scale == 1 / 500`, independent of the AOI.

    ~0.54 on the 2026-08-31 rig (1:270 map), which is exactly the ~1.85x oversize measured on
    the table, inverted.
    """
    if not ground_scale > 0:
        raise ValueError(f"ground scale must be positive, got {ground_scale!r}")
    return ground_scale / MODEL_SCALE


def place_geometry(
    local_geometry: dict[str, Any],
    center: tuple[float, float],
    rotation_degrees: float,
    scale: float = 1.0,
    offset: tuple[float, float] = (0.0, 0.0),
) -> dict[str, Any]:
    """Rotate local metres and geodesically place them around ``(longitude, latitude)``.

    `scale` (see `model_scale_factor`) is applied in the local east/north frame, before the
    rotation and before anything touches degrees: scaling and rotation about the same anchor
    commute, so the footprint keeps its size at any heading, and the anchor stays exactly
    `center` because the local frame's origin is the catalog's `bbox_center`.

    `offset` is the building's own east/north correction, in the *same* local metres as the
    geometry -- the gap between where the marker is actually glued and the block's bbox centre.
    Added before the rotation, so it turns with the block instead of with the compass: a
    correction stored in world axes would keep pointing east and be wrong the instant the
    operator turned the block, which is the one thing this frame choice exists to prevent.
    Being in catalog metres, it also shrinks onto the table with the footprint rather than
    needing its own scale rule.
    """
    center_lng, center_lat = center
    angle = math.radians(rotation_degrees)
    cosine, sine = math.cos(angle), math.sin(angle)
    offset_east, offset_north = float(offset[0]), float(offset[1])

    def to_wgs84(position: list[float]) -> list[float]:
        east = (float(position[0]) + offset_east) * scale
        north = (float(position[1]) + offset_north) * scale
        rotated_east = east * cosine - north * sine
        rotated_north = east * sine + north * cosine
        distance = math.hypot(rotated_east, rotated_north)
        if distance == 0:
            return [center_lng, center_lat]
        bearing = math.degrees(math.atan2(rotated_east, rotated_north))
        lng, lat, _back_azimuth = _GEOD.fwd(center_lng, center_lat, bearing, distance)
        return [lng, lat]

    return _map_coordinates(
        local_geometry,
        to_wgs84,
        source_key="local_coordinates",
        target_key="coordinates",
    )


def building_feature(
    building: dict[str, Any],
    marker_id: int,
    center: tuple[float, float],
    rotation: float,
    scale: float = 1.0,
    *,
    table_direction_to_map=None,
) -> dict[str, Any]:
    """The runtime geometry for one detected marker: catalog shape, placed and sized for the map.

    `scale` is the session's `model_scale_factor` (see there). The building's own
    `calibration` block (see `building_calibration_of`) then lands on top of it as the residual:

        effective_rotation = detected - marker_reference_rotation + rotation_offset_deg
        effective_scale    = session model_scale_factor * scale_residual
        offset             = (offset_east_m, offset_north_m), in the block's own frame

    That order is the error model, not a preference: the global pixel-to-geography mapping is
    applied first and the per-building constants absorb only what is left. Reversed, each
    building's offset would quietly soak up the homography error in its corner of the table and
    be wrong the moment the block was moved.

    `table_direction_to_map` is the D2 fix: a callable that takes one table-frame direction in
    degrees and returns the map-frame direction it actually points in, by pushing it through the
    session's homography at this marker's own pixel position (see
    `pixel_to_utm.direction_through_homography`). The homography is not a rotation -- the rig
    carries 0.42 degrees of shear and 3.5% anisotropy -- so an angle carried across as a bare
    scalar is wrong by up to 1.8 degrees in a way that depends on both heading and position, and
    no constant can absorb it.

    Both the detected angle *and* the stored reference go through it, which is what makes the fix
    a pure improvement rather than a convention change: the reference was recorded in table-pixel
    degrees by `build.py` (which has no homography), so converting only the detected angle would
    subtract two numbers living in different frames and inject the table's own heading as a fresh
    systematic error. Converting both at the same pixel makes the difference the real, local,
    world-frame angle between the block's current heading and its reference heading -- and leaves
    every already-registered reference valid, with no arithmetic migration.

    Left as `None` (the default) the angle is carried across as a scalar exactly as before, which
    is what keeps this module free of any homography or OpenCV knowledge and testable on its own.

    `model_scale_factor` is reported back on the feature so the projected drawing carries the
    number it was actually drawn with -- when a building comes out the wrong size on the table,
    that says whether the scale or the catalog is at fault without reading the server's log.

    `alignment_verified` is reported for D1: false means nobody has ever checked that this
    building's `marker_reference_rotations` were captured with the block facing the way the
    catalog thinks it faces, so the footprint below may be turned by a constant of up to 180
    degrees. It still draws -- it is more useful on the table with a suspect heading than absent
    -- but the projection and the panel must mark it rather than present it as measured.
    """
    reference_rotation = float(
        building.get("marker_reference_rotations", {}).get(str(marker_id), 0.0)
    )
    if table_direction_to_map is not None:
        rotation = table_direction_to_map(rotation)
        reference_rotation = table_direction_to_map(reference_rotation)
    calibration = building_calibration_of(building)
    effective_rotation = (
        rotation - reference_rotation + applied_rotation_offset(calibration) + 180.0
    ) % 360.0 - 180.0
    effective_scale = scale * calibration["scale_residual"]
    geometry = place_geometry(
        building["geometry"],
        center,
        effective_rotation,
        scale=effective_scale,
        offset=(calibration["offset_east_m"], calibration["offset_north_m"]),
    )
    return {
        "type": "Feature",
        "properties": {
            "marker_id": marker_id,
            "building_id": building["building_id"],
            "city_scope_id": building["city_scope_id"],
            "center": list(center),
            "rotation": effective_rotation,
            "model_scale_factor": effective_scale,
            "alignment_verified": alignment_is_verified(building),
            "bbox": geometry_bbox(geometry),
        },
        "geometry": geometry,
    }
