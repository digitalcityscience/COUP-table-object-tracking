r"""Export the physical-building catalog for QGIS.

Create a QGIS-readable comparison file containing only catalog buildings. Its
GeoJSON ``geometry`` is the official longitude/latitude footprint, while
``properties.local_geometry`` keeps the corresponding table-local coordinates::

    python building_catalog\build.py `
      --export-coordinates `
        ..\COUP-table-web-interface\buildings_all.geojson `
        building_catalog\physical-building-catalog-coordinates.geojson

Registration used to live here too, and it was the source of the alignment bug
fixed on 2026-09-04: this tool observed whatever heading the block happened to be
lying at and wrote it down as the building's true-north orientation. It had no way
to do better -- a terminal cannot show the operator which way the real building
faces, and the ArUco angle alone cannot supply it, since the same marker glued
straight, sideways or upside down reads identically.

Registration is now done from the frontend's "Register building" panel, which
projects the building's own footprint onto the table at its real heading so the
operator can turn the block parallel to it before confirming. `server.py`'s
``register_building`` message writes the catalog entry from that confirmation. The
interactive mode here was removed rather than left alongside it, because the two
produced indistinguishable catalog entries and only one of them was measured.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from physical_building_catalog import load_catalog


DEFAULT_SOURCE = SCRIPT_DIR / "buildings_all.geojson"
DEFAULT_OUTPUT = SCRIPT_DIR / "physical-building-catalog.json"


def load_source_buildings(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as source_file:
        collection = json.load(source_file)

    if collection.get("type") != "FeatureCollection" or not isinstance(collection.get("features"), list):
        raise ValueError(f"{path} is not a GeoJSON FeatureCollection")

    buildings: dict[str, dict[str, Any]] = {}
    for index, feature in enumerate(collection["features"]):
        building_id = feature.get("properties", {}).get("building_id")
        if not isinstance(building_id, str) or not building_id.strip():
            raise ValueError(f"Feature {index} has no building_id")
        normalized_building_id = building_id.strip().upper()
        if normalized_building_id in buildings:
            raise ValueError(f"Duplicate building_id in source: {normalized_building_id}")
        buildings[normalized_building_id] = feature
    return buildings


def export_coordinate_comparison(
    catalog_path: Path,
    official_source_path: Path,
    export_path: Path,
) -> None:
    """Export catalog entries with official and table-local geometries."""
    catalog = load_catalog(catalog_path)
    official_buildings = load_source_buildings(official_source_path)
    features: list[dict[str, Any]] = []
    missing_ids: list[str] = []

    for entry in catalog["buildings"]:
        building_id = str(entry["building_id"]).strip().upper()
        official = official_buildings.get(building_id)
        if official is None:
            missing_ids.append(building_id)
            continue

        features.append(
            {
                "type": "Feature",
                "id": building_id,
                "geometry": official.get("geometry"),
                "properties": {
                    "building_id": building_id,
                    "marker_ids": entry.get("marker_ids", []),
                    "local_bbox": entry.get("local_bbox"),
                    "local_geometry": entry.get("geometry"),
                    "catalog_source_properties": entry.get("source_properties", {}),
                    "official_properties": official.get("properties", {}),
                },
            }
        )

    if missing_ids:
        missing = ", ".join(sorted(missing_ids))
        raise ValueError(
            f"Catalog building_id values missing from {official_source_path}: {missing}"
        )

    result = {
        "type": "FeatureCollection",
        "name": "physical-building-coordinate-comparison",
        "coordinate_system": {
            "geometry": "official longitude/latitude (WGS84)",
            "properties.local_geometry": catalog.get("coordinate_system"),
        },
        "features": features,
    }
    export_path.parent.mkdir(parents=True, exist_ok=True)
    with export_path.open("w", encoding="utf-8") as export_file:
        json.dump(result, export_file, ensure_ascii=False, indent=2)
        export_file.write("\n")

    print(
        f"Wrote {len(features)} matched buildings with official and local "
        f"coordinates to {export_path}"
    )


#: Printed when this tool is run with no arguments, which is how the interactive registration
#: mode used to be started. Named and spelled out rather than left as a bare argparse error,
#: because the muscle memory it interrupts is "run build.py to register a building" and the
#: operator needs to be told where that went, not that a flag is missing.
REGISTRATION_MOVED_NOTICE = """
  Registration is no longer done here.

  This tool recorded whatever heading the block happened to be lying at and wrote it
  down as the building's true-north orientation. Nothing could have made that correct
  from a terminal: the ArUco angle cannot say how the marker is glued to the block.

  Register from the frontend's "Register building" panel instead. It projects the
  building's own footprint onto the table at its real heading; turn the block parallel
  to it (anywhere on the table -- only the angle is measured) and confirm.

  This tool still exports the catalog for QGIS:
    python building_catalog\\build.py --export-coordinates <official.geojson> <out.geojson>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the physical-building catalog alongside its official geometry"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--export-coordinates",
        nargs=2,
        type=Path,
        metavar=("OFFICIAL_GEOJSON", "OUTPUT_GEOJSON"),
        help=(
            "export --output catalog buildings with both official lon/lat and "
            "table-local geometries"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.export_coordinates:
        official_source, export_output = arguments.export_coordinates
        export_coordinate_comparison(arguments.output, official_source, export_output)
    else:
        print(REGISTRATION_MOVED_NOTICE)
