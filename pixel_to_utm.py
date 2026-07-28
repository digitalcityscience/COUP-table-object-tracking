from dataclasses import dataclass

import cv2
import numpy as np

"""
TODO server py to listen to frontend for messages
- map calibration
- building calibration
-> store in redis or database
"""


@dataclass
class BasemapCalibrationPoint:
    pixel_position: tuple[float, float]
    utm_position: tuple[float, float]


@dataclass
class BasemapHomography:
    """Homography that maps pixel coordinates to a *local* UTM frame, plus the
    offset needed to translate back to true UTM coordinates.

    The local frame exists to keep the DLT solve numerically well-conditioned:
    UTM eastings/northings are ~1e5-1e7 while pixel coordinates are ~1e2-1e3,
    and mixing such different magnitudes (especially in float32) can destabilize
    the homography solve. We instead solve pixel -> (utm - utm_offset), then add
    utm_offset back onto any projected point.
    """

    matrix: np.ndarray
    utm_offset: tuple[float, float]


def create_basemap_homography(basemap_calibration_points: list[BasemapCalibrationPoint]) -> BasemapHomography:
    """
    Calculate the homography mapping pixel coordinates to UTM coordinates from
    an arbitrary set (>= 4) of calibration points.

    Unlike the camera calibration case, the calibration points here are
    projected onto the table and do not form a known rectangle, so we can't
    build a synthetic destination rectangle. Instead we solve directly for
    the projective transform between the pixel-space points and their
    corresponding real-world UTM points using cv2.findHomography, which
    supports any planar point configuration and any number of points >= 4
    (using least-squares/RANSAC to average out noise when there are more
    than the minimum 4).

    To keep the solve numerically stable, UTM coordinates (~1e5-1e7) are
    de-meaned around their centroid before solving, since mixing them
    directly with pixel coordinates (~1e2-1e3) in a single DLT solve can be
    poorly conditioned. The centroid offset is returned alongside the
    homography so it can be added back after projecting.

    Args:
        basemap_calibration_points: >= 4 pixel/UTM point correspondences

    Returns:
        BasemapHomography: the pixel -> local-UTM matrix plus the UTM offset
        to add back to any projected point.
    """
    if len(basemap_calibration_points) < 4:
        raise ValueError("Need at least 4 calibration points to compute a homography")

    pixel_points = np.array(
        [p.pixel_position for p in basemap_calibration_points], dtype=np.float64
    )
    utm_points = np.array(
        [p.utm_position for p in basemap_calibration_points], dtype=np.float64
    )

    utm_offset = utm_points.mean(axis=0)
    local_utm_points = utm_points - utm_offset

    # method=0 -> exact/least-squares DLT solution. Use cv2.RANSAC instead if
    # calibration points may contain outliers (e.g. noisy manual clicks).
    matrix, _mask = cv2.findHomography(pixel_points, local_utm_points, method=0)

    return BasemapHomography(matrix=matrix, utm_offset=(utm_offset[0], utm_offset[1]))


def project_pixel_to_utm(homography: BasemapHomography, pixel_position: tuple[float, float]) -> tuple[float, float]:
    """Project a single pixel-space marker position into UTM using the homography."""
    point = np.array([[pixel_position]], dtype=np.float64)  # shape (1, 1, 2)
    projected = cv2.perspectiveTransform(point, homography.matrix)
    local_x, local_y = projected[0, 0]
    return (
        float(local_x) + homography.utm_offset[0],
        float(local_y) + homography.utm_offset[1],
    )


def project_pixels_to_utm(homography: BasemapHomography, pixel_positions: np.ndarray) -> np.ndarray:
    """Project many pixel-space positions into UTM in a single batched call.

    Args:
        homography: the pixel -> UTM homography.
        pixel_positions: array of shape (N, 2) of pixel (x, y) coordinates.

    Returns:
        Array of shape (N, 2) of UTM (x, y) coordinates.
    """
    if len(pixel_positions) == 0:
        return np.empty((0, 2), dtype=np.float64)

    pixel_positions = np.asarray(pixel_positions, dtype=np.float64)

    # cv2.perspectiveTransform requires points shaped (N, 1, 2) rather than (N, 2)
    points_for_cv2 = pixel_positions.reshape(-1, 1, 2)
    projected_points = cv2.perspectiveTransform(points_for_cv2, homography.matrix)
    utm_points = projected_points.reshape(-1, 2)

    return utm_points + np.asarray(homography.utm_offset, dtype=np.float64)