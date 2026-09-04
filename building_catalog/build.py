r"""Build the working physical-building catalog or export it for QGIS.

Interactive catalog creation (the existing mode)::

    python building_catalog\build.py

Create a QGIS-readable comparison file containing only catalog buildings. Its
GeoJSON ``geometry`` is the official longitude/latitude footprint, while
``properties.local_geometry`` keeps the corresponding table-local coordinates::

    python building_catalog\build.py `
      --export-coordinates `
        ..\COUP-table-web-interface\buildings_all.geojson `
        building_catalog\physical-building-catalog-coordinates.geojson

Both commands use the working catalog beside this file by default. The runtime
continues to read the project-root ``physical-building-catalog.json``; copy a
finished working catalog there manually when it is ready. Export mode only
reads files and matches buildings case-insensitively by ``building_id``.
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import sys
import threading
from collections import Counter
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2

from calibration_handler import load_calibration_markers, run_initial_calibration_if_needed
from camera_stitching import process_and_join_streams, setup_camera_transforms
from detection import detect_markers
from hud import draw_monitor_window, draw_status_window
from calibration_contract import MAP_CALIBRATION_MARKER_IDS
from marker import IGNORED_MARKER_ID, map_detected_markers
from physical_building_catalog import (
    alignment_is_verified,
    catalog_entry,
    load_catalog,
    marker_index,
    save_catalog,
)


DEFAULT_SOURCE = SCRIPT_DIR / "buildings_all.geojson"
DEFAULT_OUTPUT = SCRIPT_DIR / "physical-building-catalog.json"

#: Marker ids this tool must never assign to a building, derived from the contracts that own them
#: rather than restated. It used to hard-code `{200, 201, 202, 203, 500}`, which silently went
#: stale the moment the calibration grid grew to nine markers: registering a building while the
#: Table window is projecting them -- the normal state during a rig session -- would have decoded
#: 204-208, passed them through the sighting gate, and written a calibration marker into a
#: building's `marker_ids`. At runtime a footprint would then be drawn on top of that marker.
RESERVED_MARKER_IDS = set(MAP_CALIBRATION_MARKER_IDS) | {IGNORED_MARKER_ID}


def confirm_replace(
    building_id: str,
    existing_marker_ids: list[int],
    detected_marker_ids: list[int],
    input_fn=input,
    print_fn=print,
) -> bool:
    print_fn(f"{building_id} already exists with marker IDs {existing_marker_ids}.")
    print_fn(f"Detected marker IDs: {detected_marker_ids}.")
    return input_fn("Are you sure you want to replace it? [y/N] ").strip().lower() in {"y", "yes"}


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


def _replace_latest(
    marker_snapshots: "queue.Queue[dict[int, float]]", marker_rotations: dict[int, float]
) -> None:
    if marker_snapshots.full():
        try:
            marker_snapshots.get_nowait()
        except queue.Empty:
            pass
    marker_snapshots.put_nowait(marker_rotations)


def run_visible_detection(
    stitching_setup,
    marker_snapshots: "queue.Queue[dict[int, float]]",
    stop_event: threading.Event,
) -> None:
    """Continuously run the same stitched detection/HUD loop as server.py."""
    try:
        for stitched_image in process_and_join_streams(stitching_setup, log_progress=False):
            if stop_event.is_set():
                break
            corners, ids, rejected = detect_markers(stitched_image)
            detected = map_detected_markers("catalog", ids, corners)
            draw_monitor_window(stitched_image, corners, rejected, "catalog")
            draw_status_window(detected, "catalog")

            marker_rotations = {
                marker_id: float(marker.position[2])
                for marker_id, marker in detected.items()
                if marker_id not in RESERVED_MARKER_IDS
            }
            _replace_latest(marker_snapshots, marker_rotations)
    finally:
        cv2.destroyAllWindows()


def _circular_mean_degrees(angles: list[float]) -> float:
    sine = sum(math.sin(math.radians(angle)) for angle in angles)
    cosine = sum(math.cos(math.radians(angle)) for angle in angles)
    return math.degrees(math.atan2(sine, cosine))


def observe_marker_reference_rotations(
    marker_snapshots: "queue.Queue[dict[int, float]]", sample_count: int
) -> dict[int, float]:
    counts: Counter[int] = Counter()
    rotations: dict[int, list[float]] = {}
    for _ in range(sample_count):
        snapshot = marker_snapshots.get(timeout=5)
        counts.update(snapshot.keys())
        for marker_id, rotation in snapshot.items():
            rotations.setdefault(marker_id, []).append(rotation)

    minimum_hits = max(2, sample_count // 2)
    return {
        marker_id: _circular_mean_degrees(rotations[marker_id])
        for marker_id, hits in sorted(counts.items())
        if hits >= minimum_hits
    }


def observe_marker_ids(
    marker_snapshots: "queue.Queue[dict[int, float]]", sample_count: int
) -> list[int]:
    """Compatibility wrapper for callers interested only in stable marker IDs."""
    return sorted(observe_marker_reference_rotations(marker_snapshots, sample_count))


#: What registration actually captured, spelled out at the moment it happens.
#:
#: Registration records the heading the block *happened to be lying at*, and until 2026-09-04 the
#: system then treated that heading as the building's true-north orientation without ever saying
#: so -- the D1 bug. The maths is exact (with `detected == reference` the drawn footprint lands on
#: the catalog's real heading to within 0.000 degrees), which is precisely why the unchecked
#: assumption underneath it was invisible: every registration looked perfect and every building
#: could still be turned by up to 180 degrees.
#:
#: Printed rather than merely stored because the operator is standing at the table with the block
#: in their hand right now. This is the one moment the measurement below costs nothing.
ALIGNMENT_NOT_MEASURED_NOTICE = """
  !! {building_id}: absolute alignment NOT measured.

     Registration recorded the heading this block was lying at just now and nothing has
     checked that it is the direction the real building faces. {building_id} will draw, but
     it may be turned by a constant of up to 180 deg, and it is flagged `alignment_verified:
     false` everywhere until measured.

     To measure it, once, while the rig is up:
       1. Turn the block until the projected polygon sits on top of the real building
          in the basemap underneath it.
       2. Read the `rotation` the panel reports for {building_id} at that moment.
       3. Save `rotation_offset_deg` = -(that value) from the panel.
"""


def print_alignment_not_measured(building_id: str, print_fn=print) -> None:
    """Tell the operator, at the table, that the building they just registered is unaligned."""
    print_fn(ALIGNMENT_NOT_MEASURED_NOTICE.format(building_id=building_id))


def ensure_markers_are_unique(
    catalog: dict[str, Any], building_id: str, marker_ids: list[int]
) -> None:
    for marker_id in marker_ids:
        owner = marker_index(catalog).get(marker_id)
        if owner is not None and owner["building_id"] != building_id:
            raise ValueError(
                f"Marker(s) {[marker_id]} already belong to {owner['building_id']}; "
                "a marker may identify only one physical building"
            )


def build_catalog(args: argparse.Namespace) -> None:
    source_buildings = load_source_buildings(args.source)
    catalog = load_catalog(args.output)
    run_initial_calibration_if_needed()
    calibration_data = load_calibration_markers(str(args.camera_calibration))
    stitching_setup = setup_camera_transforms(calibration_data)
    marker_snapshots: "queue.Queue[dict[int, float]]" = queue.Queue(maxsize=max(args.samples * 2, 60))
    stop_event = threading.Event()
    detection_thread = threading.Thread(
        target=run_visible_detection,
        args=(stitching_setup, marker_snapshots, stop_event),
        name="catalog-detection-worker",
        daemon=True,
    )
    detection_thread.start()

    print(f"Loaded {len(source_buildings)} source buildings from {args.source}")
    print("Standalone camera detection is running; server.py is not needed.")
    print("The stitched camera monitor and marker status windows should now be visible.")
    print("Keep only the building being registered on the table.")

    try:
        while True:
            building_id = input("\nWhich building is on the table? (e.g. G17, or 'done'): ").strip().upper()
            if building_id in {"DONE", "QUIT", "EXIT"}:
                break
            if building_id not in source_buildings:
                print(f"Unknown building_id {building_id!r}")
                continue

            print(f"Observing {args.samples} live camera frames for {building_id}...")
            try:
                marker_reference_rotations = observe_marker_reference_rotations(
                    marker_snapshots, args.samples
                )
                marker_ids = sorted(marker_reference_rotations)
            except queue.Empty:
                print("Camera frames stopped arriving. Check camera connections and calibration.")
                continue
            if not marker_ids:
                print("No stable building marker detected. Reposition the object and try again.")
                continue

            existing = next((item for item in catalog["buildings"] if item["building_id"] == building_id), None)
            if existing is not None and not confirm_replace(building_id, existing["marker_ids"], marker_ids):
                print("Skipped; existing record was not changed.")
                continue

            try:
                ensure_markers_are_unique(catalog, building_id, marker_ids)
            except ValueError as error:
                print(error)
                continue

            entry = catalog_entry(
                source_buildings[building_id], marker_ids, marker_reference_rotations
            )
            catalog["buildings"] = [item for item in catalog["buildings"] if item["building_id"] != building_id]
            catalog["buildings"].append(entry)
            save_catalog(args.output, catalog)
            print(
                f"Saved {building_id} -> {marker_ids}, reference rotations "
                f"{marker_reference_rotations} ({len(catalog['buildings'])} catalog buildings)"
            )
            if not alignment_is_verified(entry):
                print_alignment_not_measured(building_id)
    finally:
        stop_event.set()
        detection_thread.join(timeout=2)
        cv2.destroyAllWindows()
        save_catalog(args.output, catalog)
        print(f"Catalog written to {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a single physical-building GeoJSON catalog from live table markers"
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--camera-calibration",
        type=Path,
        default=PROJECT_ROOT / "calibration_markers.json",
    )
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument(
        "--export-coordinates",
        nargs=2,
        type=Path,
        metavar=("OFFICIAL_GEOJSON", "OUTPUT_GEOJSON"),
        help=(
            "skip camera capture and export --output catalog buildings with both "
            "official lon/lat and table-local geometries"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.export_coordinates:
        official_source, export_output = arguments.export_coordinates
        export_coordinate_comparison(arguments.output, official_source, export_output)
    else:
        build_catalog(arguments)
