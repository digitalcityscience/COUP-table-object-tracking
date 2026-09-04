import math
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

#: The baseline `direction_through_homography` measures a direction over, in table pixels.
#:
#: Long enough that the float64 UTM difference carries plenty of significant digits (one pixel is
#: ~0.33 m, so ten is ~3.3 m against coordinates de-meaned to the calibrated quad), and short
#: enough that the secant still describes the homography *at* the marker rather than averaged
#: across the table. It is also about the size of a real marker edge, which is the thing the
#: angle actually came from.
DIRECTION_PROBE_PIXELS = 10.0


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


def axis_metres_per_table_pixel(
    homography: BasemapHomography, at_pixel: tuple[float, float]
) -> tuple[float, float]:
    """How many real-world metres one table pixel spans at `at_pixel`, per axis.

    Read straight off the homography as |H(x + 1, y) - H(x, y)| and |H(x, y + 1) - H(x, y)| in
    UTM metres, so it needs no knowledge of the AOI, the table's physical size, or anything the
    frontend sends.

    Returned as a pair rather than collapsed here because the two numbers are not the same and
    the difference is itself a diagnostic: on the 2026-09-04 rig they came out 0.3297 and 0.3420
    m/px, a ratio of 0.9641. A projection that is 3.5% out of square is a projector or AOI
    problem, and a caller that wants to notice it needs to see both.

    The pair is `(pixel-x, pixel-y)` -- the *table image's* axes, not the compass's. They are
    named for the pixel axes rather than east/north deliberately: this rig's table happens to sit
    within a fifth of a degree of north, so calling them east/north would read as true here and
    quietly mislabel the diagnostic on any table turned against north.
    """
    x, y = float(at_pixel[0]), float(at_pixel[1])
    origin, one_pixel_x, one_pixel_y = project_pixels_to_utm(
        homography, np.array([[x, y], [x + 1.0, y], [x, y + 1.0]], dtype=np.float64)
    )
    return (
        float(np.hypot(*(one_pixel_x - origin))),
        float(np.hypot(*(one_pixel_y - origin))),
    )


def project_utm_to_pixels(homography: BasemapHomography, utm_positions: np.ndarray) -> np.ndarray:
    """The inverse of `project_pixels_to_utm`: UTM back to table pixels.

    Needed because registration has to answer "which marker is on the target?", and the target is
    a geographic position while marker sightings are table pixels. Comparing them in pixels keeps
    the tolerance in the unit the question is actually about -- one table pixel is one millimetre,
    so "within 10 cm of the target" is a number anyone can check against a physical block.
    """
    if len(utm_positions) == 0:
        return np.empty((0, 2), dtype=np.float64)
    inverse = np.linalg.inv(homography.matrix)
    local = np.asarray(utm_positions, dtype=np.float64) - np.asarray(
        homography.utm_offset, dtype=np.float64
    )
    projected = cv2.perspectiveTransform(local.reshape(-1, 1, 2), inverse)
    return projected.reshape(-1, 2)


def metres_per_table_pixel(
    homography: BasemapHomography, at_pixel: tuple[float, float]
) -> float:
    """How many real-world metres one table pixel spans at `at_pixel`, over both axes.

    The *geometric* mean of the two axis scales (see `axis_metres_per_table_pixel`), not the x
    axis alone. This used to probe only pixel-x, on the stated grounds that "the AOI is
    aspect-locked to the table by construction, so the two axes carry the same scale". Measured on
    the real rig that claim is false: |dx|/|dy| = 0.9641, a 3.5% disagreement, which made
    `ground_scale` -- and therefore every building's `model_scale_factor` -- systematically wrong
    by ~1.8% against the mean and by up to 3.6% against whichever axis the operator happened to
    be measuring with a ruler.

    Geometric rather than arithmetic mean because this number is used as a single isotropic
    factor for a two-dimensional drawing: sqrt(|dx| * |dy|) is the side of the square with the
    same area as the real dx-by-dy parallelogram, so a footprint drawn with it comes out the
    right *size* even though a 3.5%-anisotropic map cannot make it the right size on both axes at
    once. An arithmetic mean would leave the area systematically large.
    """
    x_metres, y_metres = axis_metres_per_table_pixel(homography, at_pixel)
    return float(math.sqrt(x_metres * y_metres))


def direction_through_homography(
    homography: BasemapHomography,
    at_pixel: tuple[float, float],
    table_direction_degrees: float,
    probe_pixels: float = DIRECTION_PROBE_PIXELS,
) -> float:
    """Where a table-frame direction at `at_pixel` actually points on the ground, in degrees.

    Both angles are counter-clockwise-positive: `table_direction_degrees` is measured from the
    table's +x pixel axis (what `detection.normalizeCorners` reports), and the result is measured
    from UTM east, which is the frame `physical_building_catalog.place_geometry` rotates in.

    This is the missing call, not missing data. The pipeline projects marker *points* through the
    homography and then copies the marker's *angle* across as a bare `atan2` scalar -- correct
    only if the map were a pure similarity transform of the table. It is not: the 2026-09-04 rig
    measured 0.42 degrees of axis-squareness error and 3.5% anisotropy, which together move a
    carried-across angle by up to 1.8 degrees, varying with both heading and position, so no
    constant `rotation_offset_deg` can absorb it. Pushing two points through instead of one
    number closes the sign question, the table-to-north tilt, the shear and the anisotropy at
    once, and leaves no magic constant behind: the answer is whatever the homography says.

    Implemented as a finite difference over `probe_pixels` rather than by differentiating the
    matrix because a homography is only locally linear -- the secant over a short baseline is the
    direction a *real* marker edge of that length would map to, which is the thing being asked
    about.
    """
    x, y = float(at_pixel[0]), float(at_pixel[1])
    angle = math.radians(float(table_direction_degrees))
    origin, probe = project_pixels_to_utm(
        homography,
        np.array(
            [
                [x, y],
                [x + probe_pixels * math.cos(angle), y + probe_pixels * math.sin(angle)],
            ],
            dtype=np.float64,
        ),
    )
    east, north = probe - origin
    return math.degrees(math.atan2(float(north), float(east)))


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