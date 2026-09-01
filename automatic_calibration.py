"""Automatic version of the legacy camera-by-camera calibration workflow."""

import json
from pathlib import Path
import time

import cv2
import numpy as np

from camera import poll_frame_data
from detection import detect_markers
from image import buffer_to_array, sharpen_and_rotate_image
from rig_config import CORNER_ORDER, build_fixed_camera_setup
from save_calibration_markers import export_pictures_for_debugging
from distortion_analysis import analyze_camera_distortion


def _marker_center(marker_corners) -> list[float]:
    points = np.asarray(marker_corners).reshape(-1, 2)
    return [float(np.mean(points[:, 0])), float(np.mean(points[:, 1]))]


def calibrate_fixed_rig(output_path: Path, *, timeout_seconds: float = 60.0) -> dict:
    """Find one fixed marker at a time, exactly like the legacy manual flow."""
    camera_setup = build_fixed_camera_setup()
    frame_iterator = poll_frame_data()
    detected_markers = {camera_id: {} for camera_id in camera_setup}
    best_frames = {}

    print("Starting automatic camera-by-camera calibration...")
    print("Each target advances immediately after its first successful detection.")

    try:
        for camera_id, camera_config in camera_setup.items():
            print(f"\nCamera {camera_id} ({camera_config['position']})")
            window_name = f"Camera {camera_id}"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 960, 600)
            resolution_reported = False

            for corner in CORNER_ORDER:
                marker_id = int(camera_config["calibration_markers"][corner]["id"])
                deadline = time.monotonic() + timeout_seconds
                observed_ids = set()
                print(f"  Looking for marker {marker_id} ({corner})...")

                while True:
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
                    target_center = None

                    if ids is not None:
                        detected_ids = [
                            int(value) for value in np.asarray(ids).flatten()
                        ]
                        observed_ids.update(detected_ids)
                        cv2.aruco.drawDetectedMarkers(
                            preview, detected_corners_list, ids
                        )
                        for detected_id, detected_corners in zip(
                            detected_ids, detected_corners_list
                        ):
                            if detected_id == marker_id:
                                target_center = _marker_center(detected_corners)
                                break

                    cv2.putText(
                        preview,
                        f"Target: {marker_id} ({corner})",
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

                    if target_center is not None:
                        camera_config["calibration_markers"][corner][
                            "pixel_position"
                        ] = target_center
                        detected_markers[camera_id][corner] = {
                            "id": str(marker_id),
                            "position": target_center,
                        }
                        print(
                            f"  READ marker {marker_id}: pixel center "
                            f"({target_center[0]:.1f}, {target_center[1]:.1f})"
                        )
                        break

                    if time.monotonic() >= deadline:
                        raise RuntimeError(
                            f"Calibration timed out for camera {camera_id}, marker "
                            f"{marker_id} ({corner}). IDs observed while waiting: "
                            f"{sorted(observed_ids)}"
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
