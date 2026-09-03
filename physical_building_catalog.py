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

#: What an uncalibrated building means. A true no-op, not an approximate one: the three buildings
#: registered before this step existed carry no calibration block at all and must keep drawing
#: byte-identically.
DEFAULT_BUILDING_CALIBRATION = {
    "rotation_offset_deg": 0.0,
    "offset_east_m": 0.0,
    "offset_north_m": 0.0,
    "scale_residual": 1.0,
}

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


def calibration_from_message(payload: dict[str, Any]) -> dict[str, float]:
    """The catalog-unit calibration a `building_calibration` message asks for.

    Only the fields the message actually carries, so a save of the one axis the operator touched
    stays a partial update (see `apply_building_calibration`). Millimetre fields are converted
    to catalog metres on the way through; degrees and the scale ratio pass straight along.
    """
    calibration: dict[str, float] = {}
    for message_field, catalog_field in BUILDING_CALIBRATION_MESSAGE_FIELDS.items():
        if message_field not in payload:
            continue
        value = float(payload[message_field])
        calibration[catalog_field] = (
            table_millimetres_to_local_metres(value) if message_field.endswith("_mm") else value
        )
    return calibration


def calibration_as_message_fields(building: dict[str, Any]) -> dict[str, float]:
    """One building's stored calibration, back in the units a `building_calibration` speaks.

    The exact inverse of `calibration_from_message`, and published on every runtime feature so the
    admin panel can open showing where a building actually stands rather than at zero. Without it
    the panel has no way to read what Python already holds, and a second save built from a neutral
    draft would silently replace the first sitting's measurements instead of refining them.

    Named field-for-field after the message, so the value the panel receives is the value it sends
    back with no conversion of its own -- which is what keeps `MODEL_SCALE` a Python-only number.
    """
    stored = building_calibration_of(building)
    return {
        message_field: (
            local_metres_to_table_millimetres(stored[catalog_field])
            if message_field.endswith("_mm")
            else stored[catalog_field]
        )
        for message_field, catalog_field in BUILDING_CALIBRATION_MESSAGE_FIELDS.items()
    }


def local_metres_to_table_millimetres(metres: float) -> float:
    """The inverse of `table_millimetres_to_local_metres` — catalog metres back to table mm."""
    return metres * 1000 / MODEL_SCALE


def building_calibration_of(building: dict[str, Any]) -> dict[str, float]:
    """One building's calibration, defaulted field by field.

    Defaulted per field rather than per block, so a catalog entry that has only ever had its
    rotation nudged still reports a neutral offset and scale instead of nothing.
    """
    stored = building.get("calibration") or {}
    return {
        field: float(stored.get(field, DEFAULT_BUILDING_CALIBRATION[field]))
        for field in BUILDING_CALIBRATION_FIELDS
    }


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

    `model_scale_factor` is reported back on the feature so the projected drawing carries the
    number it was actually drawn with -- when a building comes out the wrong size on the table,
    that says whether the scale or the catalog is at fault without reading the server's log.
    """
    reference_rotation = float(
        building.get("marker_reference_rotations", {}).get(str(marker_id), 0.0)
    )
    calibration = building_calibration_of(building)
    effective_rotation = (
        rotation - reference_rotation + calibration["rotation_offset_deg"] + 180.0
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
            "bbox": geometry_bbox(geometry),
        },
        "geometry": geometry,
    }
