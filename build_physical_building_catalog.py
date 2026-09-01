"""Interactively bind physical table markers to source building footprints.

Run the tracking server in web mode, leave only one physical building on the
table, then enter that building's ``building_id`` (for example ``G07``).  The
script starts the same visible stitched-camera detection loop as server.py,
observes ArUco marker IDs, and writes the standalone physical-building catalog.
"""

from __future__ import annotations

import argparse
import json
import queue
import threading
from collections import Counter
from pathlib import Path

import cv2

from calibration_handler import load_calibration_markers, run_initial_calibration_if_needed
from camera_stitching import process_and_join_streams, setup_camera_transforms
from detection import detect_markers
from hud import draw_monitor_window, draw_status_window
from marker import map_detected_markers
from physical_building_catalog import catalog_entry, load_catalog, marker_index, save_catalog
import json


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE = SCRIPT_DIR.parent / "COUP-table-web-interface" / "buildings_all.geojson"
DEFAULT_OUTPUT = SCRIPT_DIR / "physical-building-catalog.json"
RESERVED_MARKER_IDS = {200, 201, 202, 203, 500}


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
        if building_id in buildings:
            raise ValueError(f"Duplicate building_id in source: {building_id}")
        buildings[building_id.upper()] = feature
    return buildings


def _replace_latest(marker_snapshots: "queue.Queue[set[int]]", marker_ids: set[int]) -> None:
    if marker_snapshots.full():
        try:
            marker_snapshots.get_nowait()
        except queue.Empty:
            pass
    marker_snapshots.put_nowait(marker_ids)


def run_visible_detection(
    stitching_setup,
    marker_snapshots: "queue.Queue[set[int]]",
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

            marker_ids = set()
            if ids is not None:
                marker_ids = {int(marker_id) for marker_id in ids.flatten()} - RESERVED_MARKER_IDS
            _replace_latest(marker_snapshots, marker_ids)
    finally:
        cv2.destroyAllWindows()


def observe_marker_ids(
    marker_snapshots: "queue.Queue[set[int]]", sample_count: int
) -> list[int]:
    counts: Counter[int] = Counter()
    for _ in range(sample_count):
        counts.update(marker_snapshots.get(timeout=5))

    minimum_hits = max(2, sample_count // 2)
    return sorted(marker_id for marker_id, hits in counts.items() if hits >= minimum_hits)


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
    marker_snapshots: "queue.Queue[set[int]]" = queue.Queue(maxsize=max(args.samples * 2, 60))
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
                marker_ids = observe_marker_ids(marker_snapshots, args.samples)
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

            entry = catalog_entry(source_buildings[building_id], marker_ids)
            catalog["buildings"] = [item for item in catalog["buildings"] if item["building_id"] != building_id]
            catalog["buildings"].append(entry)
            save_catalog(args.output, catalog)
            print(f"Saved {building_id} -> {marker_ids} ({len(catalog['buildings'])} catalog buildings)")
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
        default=SCRIPT_DIR / "calibration_markers.json",
    )
    parser.add_argument("--samples", type=int, default=30)
    return parser.parse_args()


if __name__ == "__main__":
    build_catalog(parse_args())
