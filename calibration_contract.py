"""The one place the map-calibration marker contract with the TOSCA-2 frontend lives.

TOSCA-2 projects four ArUco markers at the table's corners and reads them straight back
out of this server's raw marker snapshot to build its `map_calibration` handshake. The id
-> corner assignment below is that contract; it is mirrored verbatim on the frontend in
`src/collab/stores/collabTracking.ts::MAP_CALIBRATION_MARKERS` and in the vanilla
reference app's `js/state.js::MARKER_ID_TO_KEY`. Nothing else in this repo may declare
these ids.

The `4x4_1000-{id}.svg` filenames the frontend serves are not a dictionary mismatch:
OpenCV's 4x4 dictionaries are nested, so ids 200-203 are bit-identical under the
`DICT_4X4_250` that `detection.py` loads.
"""

from typing import Dict, Final, Sequence, Tuple

Corner = str
Point = Sequence[float]

#: Table-pixel density of the stitched, rectified table image: the `scale_factor` that
#: `camera_stitching.calculate_perspective_transform` warps every camera frame onto. It lives
#: here, next to the marker ids, because it is equally part of the frontend contract -- TOSCA-2
#: restates it as `collabCalibration.ts::DEFAULT_COLLAB_TABLE_CONFIG.tablePixelSpace.pixelsPerCm`
#: and every table-pixel coordinate crossing the wire is denominated in it. It is also the bridge
#: from `pixel_to_utm.metres_per_table_pixel` to a ground scale: metres per pixel only becomes a
#: map scale once you know how many pixels a table centimetre is. Nothing else may declare it.
TABLE_PIXELS_PER_CM: Final[int] = 10

#: Corner names in the fixed order `camera_stitching.calculate_perspective_transform`
#: reads them: a closed traversal of the quad, not an arbitrary set.
CORNER_ORDER: Final[Tuple[Corner, ...]] = (
    "top_left",
    "top_right",
    "bottom_right",
    "bottom_left",
)

#: id -> table corner, explicit and total. Iteration order is never relied on anywhere;
#: every consumer looks a corner up by id or walks `CORNER_ORDER`.
MAP_CALIBRATION_MARKER_CORNERS: Final[Dict[int, Corner]] = {
    200: "top_left",
    201: "top_right",
    202: "bottom_left",
    203: "bottom_right",
}

#: The four ids above, as a set, for membership tests on a detected-marker snapshot.
MAP_CALIBRATION_MARKER_IDS: Final[frozenset] = frozenset(MAP_CALIBRATION_MARKER_CORNERS)


def corner_for_map_calibration_marker(marker_id: int) -> Corner:
    """The table corner `marker_id` occupies, or raise if it is not a calibration marker."""
    try:
        return MAP_CALIBRATION_MARKER_CORNERS[int(marker_id)]
    except KeyError:
        raise KeyError(
            f"{marker_id} is not a map-calibration marker; expected one of "
            f"{sorted(MAP_CALIBRATION_MARKER_IDS)}"
        ) from None


def signed_area(points: Sequence[Point]) -> float:
    """Shoelace signed area of a polygon. Sign encodes traversal orientation."""
    total = 0.0
    count = len(points)
    for index in range(count):
        x0, y0 = points[index][0], points[index][1]
        x1, y1 = points[(index + 1) % count][0], points[(index + 1) % count][1]
        total += x0 * y1 - x1 * y0
    return total / 2.0


def is_mirrored(pixel_quad: Sequence[Point]) -> bool:
    """Whether `pixel_quad`, read in `CORNER_ORDER`, is wound the wrong way round.

    In image coordinates (y grows downward) a genuine top_left -> top_right ->
    bottom_right -> bottom_left traversal is clockwise on screen, which the shoelace
    formula reports as a positive area. A negative area means the four corner *labels*
    disagree with where the markers actually are, so the perspective transform built from
    them flips the table image.

    That flip is silent and total: ArUco codes are rotation-invariant but not
    mirror-invariant, so a mirrored stitched frame decodes almost nothing, and what it
    does decode comes back under the wrong id. This predicate is what turns that into a
    startup error instead of a camera that simply never sees the calibration markers.
    """
    return signed_area(pixel_quad) < 0


def assert_not_mirrored(pixel_quad: Sequence[Point], *, context: str) -> None:
    """Raise `ValueError` if `pixel_quad`'s corner labels imply a mirrored table image."""
    if is_mirrored(pixel_quad):
        raise ValueError(
            f"{context}: the four calibration markers are labelled in mirrored order "
            f"(signed area {signed_area(pixel_quad):.1f} < 0). Read in CORNER_ORDER "
            f"{CORNER_ORDER} their pixel positions were {[list(p[:2]) for p in pixel_quad]}. "
            "A transform built from these flips the table image, and ArUco cannot decode "
            "mirrored markers -- detection would silently return almost nothing. Re-run "
            "the automatic calibration so each corner label is assigned from the marker's "
            "observed position."
        )
