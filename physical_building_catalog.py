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


def catalog_entry(feature: dict[str, Any], marker_ids: list[int]) -> dict[str, Any]:
    properties = copy.deepcopy(feature.get("properties", {}))
    building_id = str(properties.get("building_id", "")).strip().upper()
    if not building_id:
        raise ValueError("Source feature has no building_id")
    city_scope_id = properties.get("city_scope_id")
    if not isinstance(city_scope_id, str) or not city_scope_id:
        raise ValueError(f"Building {building_id} has no city_scope_id")
    geometry, local_bbox = normalize_geometry(feature["geometry"])
    return {
        "building_id": building_id,
        "city_scope_id": city_scope_id,
        "marker_ids": sorted(set(marker_ids)),
        "local_bbox": local_bbox,
        "geometry": geometry,
        "source_properties": properties,
    }


def empty_catalog() -> dict[str, Any]:
    return {"version": CATALOG_VERSION, "coordinate_system": copy.deepcopy(COORDINATE_SYSTEM), "buildings": []}


def load_catalog(path: Path) -> dict[str, Any]:
    if not path.exists():
        return empty_catalog()
    with path.open("r", encoding="utf-8") as catalog_file:
        catalog = json.load(catalog_file)
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


def place_geometry(
    local_geometry: dict[str, Any], center: tuple[float, float], rotation_degrees: float
) -> dict[str, Any]:
    """Rotate local metres and geodesically place them around ``(longitude, latitude)``."""
    center_lng, center_lat = center
    angle = math.radians(rotation_degrees)
    cosine, sine = math.cos(angle), math.sin(angle)

    def to_wgs84(position: list[float]) -> list[float]:
        east, north = float(position[0]), float(position[1])
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
    building: dict[str, Any], marker_id: int, center: tuple[float, float], rotation: float
) -> dict[str, Any]:
    geometry = place_geometry(building["geometry"], center, rotation)
    return {
        "type": "Feature",
        "properties": {
            "marker_id": marker_id,
            "building_id": building["building_id"],
            "city_scope_id": building["city_scope_id"],
            "center": list(center),
            "rotation": rotation,
            "bbox": geometry_bbox(geometry),
        },
        "geometry": geometry,
    }
