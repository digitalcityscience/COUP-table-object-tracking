from dataclasses import dataclass

import cv2
import numpy as np
from pyproj import Transformer

from calibration_contract import TABLE_PIXELS_PER_CM

"""
TODO server py to listen to frontend for messages
- map calibration
- building calibration
-> store in redis or database
"""

# EPSG:25832 = ETRS89 / UTM zone 32N, which covers Hamburg/Germany.
_WGS84_TO_UTM = Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)


@dataclass
class BasemapCalibrationPoint:
    pixel_position: tuple[float, float]
    lat_lon_position: tuple[float, float]
    utm_position: tuple[float, float] = None

    def __init__(self, pixel_position: tuple[float, float], lat_lon_position: tuple[float, float]):
        self.pixel_position = pixel_position
        self.lat_lon_position = lat_lon_position
        self.utm_position = self.latlon_to_utm(lat_lon_position)

    def latlon_to_utm(self, lat_lon_position: tuple[float, float]) -> tuple[float, float]:
        """Convert a (lat, lon) position to UTM coordinates (EPSG:25832)."""
        lat, lon = lat_lon_position
        easting, northing = _WGS84_TO_UTM.transform(lon, lat)
        return easting, northing
    

@dataclass
class BasemapHomography:
    """Homography that maps pixel coordinates to a *local* UTM frame, plus the
    offset needed to translate back to true UTM coordinates.

    The local frame exists to keep the DLT solve numerically well-conditioned:
    UTM eastings/northings are ~1e5-1e7 while pixel coordinates are ~1e2-1e3,
    and mixing such different magnitudes (especially in float32) can destabilize
    the homography solve. We instead solve pixel -> (utm - utm_offset), then add
    utm_offset back onto any projected point.

    `pixel_centroid` is the mean pixel position of the calibration points the homography was
    solved from. A projective transform has no single scale -- it stretches differently in
    different parts of the image -- so anything that needs *the* scale of this calibration
    (`ground_scale`) has to name a sample point. Carrying it on the homography pins that choice
    to the calibration itself rather than leaving each caller to invent one.
    """

    matrix: np.ndarray
    utm_offset: tuple[float, float]
    pixel_centroid: tuple[float, float]


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

    pixel_centroid = pixel_points.mean(axis=0)

    return BasemapHomography(
        matrix=matrix,
        utm_offset=(utm_offset[0], utm_offset[1]),
        pixel_centroid=(float(pixel_centroid[0]), float(pixel_centroid[1])),
    )


def project_pixel_to_utm(homography: BasemapHomography, pixel_position: tuple[float, float]) -> tuple[float, float]:
    """Project a single pixel-space marker position into UTM using the homography."""
    point = np.array([[pixel_position]], dtype=np.float64)  # shape (1, 1, 2)
    projected = cv2.perspectiveTransform(point, homography.matrix)
    local_x, local_y = projected[0, 0]
    return (
        float(local_x) + homography.utm_offset[0],
        float(local_y) + homography.utm_offset[1],
    )


def metres_per_table_pixel(
    homography: BasemapHomography, at_pixel: tuple[float, float]
) -> float:
    """How many real-world metres one table pixel spans at `at_pixel`.

    Read straight off the homography as |H(x + 1, y) - H(x, y)| in UTM metres, so it needs no
    knowledge of the AOI, the table's physical size, or anything the frontend sends. Measured
    along the pixel-x axis: the AOI is aspect-locked to the table by construction
    (`collabScenario.viewfinderScreenCorners`), so the two axes carry the same scale, and a
    single-axis probe keeps the number one thing rather than an average of two.
    """
    x, y = float(at_pixel[0]), float(at_pixel[1])
    origin, one_pixel_east = project_pixels_to_utm(
        homography, np.array([[x, y], [x + 1.0, y]], dtype=np.float64)
    )
    return float(np.hypot(*(one_pixel_east - origin)))


def ground_scale(homography: BasemapHomography) -> float:
    """The table map's scale denominator: real-world centimetres per table centimetre.

    ~270 on the 2026-08-31 rig, i.e. a 1:270 map. This is the number the physical blocks
    disagree with -- they are milled at 1:500 -- and it is derived here rather than sent by the
    frontend or measured with a ruler, so there is a single place for it to be wrong.

    Sampled at the homography's own `pixel_centroid`: near the middle of the calibrated quad,
    where a projective transform's local scale is most representative of the whole table.
    """
    metres_per_pixel = metres_per_table_pixel(homography, homography.pixel_centroid)
    return metres_per_pixel * TABLE_PIXELS_PER_CM * 100


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