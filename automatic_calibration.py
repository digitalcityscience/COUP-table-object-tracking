"""Automatic version of the legacy camera-by-camera calibration workflow.

The legacy flow (`save_calibration_markers`) asked the operator, corner by corner, which
marker id was physically there, then watched the stream until that id showed up. This flow
removes the typing without removing the measurement: it watches each camera until it has
seen four of the rig's markers, then derives the corner roles from where they actually were
(`rig_config.assign_corners_by_geometry`).

Nothing here assumes which ids a camera owns or which corner an id occupies. The previous
version assumed both -- a fixed id -> corner table, clockwise from the top-left -- and
recorded each marker at whatever corner that table claimed. The markers are laid clockwise
from the bottom-left, so every corner label came out rotated 180 degrees, the perspective
transform built from them mirrored the table image, and ArUco -- rotation-invariant but not
mirror-invariant -- stopped decoding the projected calibration markers entirely.

Because the assignment is purely geometric, the rig can be re-laid (it has been: from an
interleaved arrangement to one sequential block per table) without touching this code.
"""

import json
from pathlib import Path
import time

import cv2
import numpy as np

from calibration_contract import CORNER_ORDER
from camera import poll_frame_data
from detection import detect_markers
from image import buffer_to_array, sharpen_and_rotate_image
from rig_config import (
    MARKERS_PER_CAMERA,
    RIG_MARKER_IDS,
    assign_corners_by_geometry,
    build_fixed_camera_setup,
)
from save_calibration_markers import export_pictures_for_debugging
from distortion_analysis import analyze_camera_distortion


def _marker_center(marker_corners) -> list:
    points = np.asarray(marker_corners).reshape(-1, 2)
    return [float(np.mean(points[:, 0])), float(np.mean(points[:, 1]))]


def calibrate_fixed_rig(output_path: Path, *, timeout_seconds: float = 60.0) -> dict:
    """Measure every rig marker, then assign corners from the observed geometry."""
    camera_setup = build_fixed_camera_setup()
    frame_iterator = poll_frame_data()
    detected_markers = {camera_id: {} for camera_id in camera_setup}
    best_frames = {}

    print("Starting automatic camera-by-camera calibration...")
    print("Each camera advances once all four of its rig markers have been seen.")

    try:
        for camera_id, camera_config in camera_setup.items():
            print(f"\nCamera {camera_id} ({camera_config['position']})")
            print(
                f"  Waiting for any {MARKERS_PER_CAMERA} of the rig markers "
                f"{sorted(RIG_MARKER_IDS)}..."
            )
            window_name = f"Camera {camera_id}"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 960, 600)
            resolution_reported = False

            found: dict = {}
            observed_ids = set()
            deadline = time.monotonic() + timeout_seconds

            while len(found) < MARKERS_PER_CAMERA:
                frame_camera_id, image_data = next(frame_iterator)
                if frame_camera_id != camera_id:
                    continue

                image = sharpen_and_rotate_image(buffer_to_array(image_data))
                best_frames[camera_id] = image.copy()
                if not resolution_reported:
                    height, width = image.shape[:2]
                    print(f"  Full raw IR frame: {width}x{height} pixels")
                    resolution_reported = True

                detected_corners_list, ids, _ = detect_markers(image)
                preview = image.copy()
                detected_ids = []

                if ids is not None:
                    detected_ids = [int(value) for value in np.asarray(ids).flatten()]
                    observed_ids.update(detected_ids)
                    cv2.aruco.drawDetectedMarkers(preview, detected_corners_list, ids)
                    for detected_id, detected_corners in zip(
                        detected_ids, detected_corners_list
                    ):
                        if detected_id in RIG_MARKER_IDS and detected_id not in found:
                            center = _marker_center(detected_corners)
                            found[detected_id] = center
                            print(
                                f"  READ marker {detected_id}: pixel center "
                                f"({center[0]:.1f}, {center[1]:.1f})"
                            )

                    # More than four rig markers in view means this camera can see the
                    # neighbouring table's set too, so "whichever four" would be a coin
                    # flip. Say so rather than calibrate against an arbitrary subset.
                    if len(found) > MARKERS_PER_CAMERA:
                        raise RuntimeError(
                            f"Camera {camera_id} sees {len(found)} rig markers "
                            f"({sorted(found)}), expected exactly {MARKERS_PER_CAMERA}. "
                            "Mask or move the markers that belong to the other table."
                        )

                cv2.putText(
                    preview,
                    f"Seen {len(found)}/{MARKERS_PER_CAMERA}: {sorted(found) or 'none'}",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    255,
                    2,
                )
                cv2.putText(
                    preview,
                    f"Current frame: {', '.join(map(str, detected_ids)) or 'none'}",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    255,
                    2,
                )
                cv2.imshow(window_name, preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    raise RuntimeError("Automatic calibration was cancelled")

                if len(found) < MARKERS_PER_CAMERA and time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"Calibration timed out for camera {camera_id}: saw "
                        f"{len(found)} of {MARKERS_PER_CAMERA} rig markers "
                        f"({sorted(found)}). IDs observed while waiting: "
                        f"{sorted(observed_ids)}"
                    )

            # Corner roles come from the measurement, never from an assumed id ordering.
            corner_ids = assign_corners_by_geometry(
                found, context=f"camera {camera_id}"
            )
            for corner in CORNER_ORDER:
                marker_id = corner_ids[corner]
                center = found[marker_id]
                camera_config["calibration_markers"][corner]["id"] = str(marker_id)
                camera_config["calibration_markers"][corner]["pixel_position"] = center
                detected_markers[camera_id][corner] = {
                    "id": str(marker_id),
                    "position": center,
                }
            print(
                "  Corner assignment from observed geometry: "
                + ", ".join(f"{corner}={corner_ids[corner]}" for corner in CORNER_ORDER)
            )
    finally:
        cv2.destroyAllWindows()

    output_path.write_text(json.dumps(camera_setup, indent=2), encoding="utf-8")
    print(f"Saved fresh calibration to {output_path}")
    visualization_dir = output_path.parent / "calibration_visualizations"
    export_pictures_for_debugging(
        detected_markers, best_frames, output_dir=str(visualization_dir)
    )
    analyze_camera_distortion(camera_setup, output_dir=str(visualization_dir))
    print(f"Regenerated calibration visualizations in {visualization_dir}")
    return camera_setup
