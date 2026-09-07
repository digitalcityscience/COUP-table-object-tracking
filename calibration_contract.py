"""The one place the map-calibration marker contract with the TOSCA-2 frontend lives.

TOSCA-2 projects ArUco markers into the AOI and reads them straight back out of this
server's raw marker snapshot to build its `map_calibration` handshake. Two groups: the
four corner ids (200-203), whose id -> corner assignment is a physical contract mirrored
verbatim on the frontend in `src/collab/stores/collabTracking.ts::MAP_CALIBRATION_MARKERS`
and in the vanilla reference app's `js/state.js::MARKER_ID_TO_KEY`; and the six extra grid
ids (206, 207, 209-212), which this server admits but does not name. Nothing else in this
repo may declare these ids.

The `4x4_1000-{id}.svg` filenames the frontend serves are not a dictionary mismatch:
OpenCV's 4x4 dictionaries are nested, so these ids are bit-identical under the
`DICT_4X4_250` that `detection.py` loads.

Ids 204, 205 and 208 were retired 2026-09-07: they sat at the grid's exact horizontal centre,
which on a table built from two desks pushed together is exactly where the camera-stitch seam
runs, so a marker projected there straddled the seam and could never be decoded. 204 and 205
were replaced by a left/right pair straddling the seam from a safe distance instead of sitting
on it (frontend `MapCalibrationMarkerBand`'s `midLeft`/`midRight`) — see
`TOSCA-2/public/collab/calibration-markers/README.md`. 208's replacement pair (213/214) shipped
the same day and both halves are gone again: 214 (the midRight half) was retired within hours
because that row sits at the AOI's vertical centre too, which on the live rig is where the
ceiling beamer's projection is brightest, and a marker there is overexposed and undecodable
regardless of its distance from the seam. 213 read fine and was kept -- but keeping it alone
left the centre row with two points on the left half (206, 213) and one on the right (207).

That asymmetry was retired the same day, second rig session ("yamuk binalar"): buildings drawn
uniformly crooked across the whole table. `create_basemap_homography` solves least squares over
every correspondence TOSCA-2 sends, so an unbalanced point cloud does not let per-marker
detection error cancel -- it biases the solve, and a biased homography fails as a shear. The id
list below is therefore deliberately left-right symmetric about the seam, and must stay that
way: a midLeft id added here without its midRight partner reintroduces that bug.
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

#: The six extra map-calibration marker ids TOSCA-2 projects alongside the four corners: the
#: left/right edge midpoints plus two seam-clearance pairs over the projectable area (workflow
#: step 5, `collabTracking.ts::MAP_CALIBRATION_MARKERS`).
#:
#: They exist because a four-point homography has no freedom left: `cv2.findHomography` passes
#: through all four exactly and dumps every bit of detection noise into the map everywhere else.
#: With more correspondences `create_basemap_homography` solves least-squares instead, so the
#: noise averages out and there is a residual to look at. Nothing in the homography code changes
#: to accept them -- it has taken `>= 4` points from the start.
#:
#: 209/210 and 211/212 are full pairs replacing two of what used to be three single markers sitting
#: dead-centre (204, 205, 208) -- on a table assembled from two desks pushed together that centre
#: line is exactly the camera-stitch seam, so those three ids were never decodable on that hardware
#: (2026-09-07). Each pair straddles the seam from a safe distance instead.
#:
#: 208's pair was 213/214, straddling the same seam on the vertical centre row -- both are gone.
#: 214 (the midRight half) was retired hours after it shipped: that row is also exactly where the
#: ceiling beamer's projection is brightest, and 214's side of it proved undecodable no matter the
#: seam clearance. 213 read fine, but a lone midLeft with no midRight partner made the point cloud
#: asymmetric and sheared the homography (see this module's docstring), so it went too. The centre
#: row is back to just its two edge midpoints, 206 and 207.
#:
#: This tuple must stay left-right symmetric about the seam. Adding a midLeft id without its
#: midRight partner is the "yamuk binalar" bug.
#:
#: They are declared here for one reason: `marker.py::reduceToCalibrationMarkers` waves calibration
#: markers past the confidence gate, and an id missing from this contract would be held back by it
#: and never reach the frontend to be read. Their id -> place mapping is TOSCA-2's business, not
#: this server's, so only the ids appear here.
EXTRA_MAP_CALIBRATION_MARKER_IDS: Final[Tuple[int, ...]] = (206, 207, 209, 210, 211, 212)

#: Every map-calibration id -- the four corners plus the grid -- as a set, for membership tests on
#: a detected-marker snapshot.
MAP_CALIBRATION_MARKER_IDS: Final[frozenset] = frozenset(
    set(MAP_CALIBRATION_MARKER_CORNERS) | set(EXTRA_MAP_CALIBRATION_MARKER_IDS)
)


def corner_for_map_calibration_marker(marker_id: int) -> Corner:
    """The table corner `marker_id` occupies, or raise if it does not occupy one.

    Only the four corner ids have a corner. The grid ids in
    `EXTRA_MAP_CALIBRATION_MARKER_IDS` are calibration markers too, but they sit at edge midpoints
    and the centre, which this server has no need to name.
    """
    try:
        return MAP_CALIBRATION_MARKER_CORNERS[int(marker_id)]
    except KeyError:
        raise KeyError(
            f"{marker_id} does not occupy a table corner; expected one of "
            f"{sorted(MAP_CALIBRATION_MARKER_CORNERS)}"
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
