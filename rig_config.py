"""Static physical configuration for the two-table camera rig.

The rig was deliberately re-laid-out so each table owns a contiguous, sequential block of
marker ids (863 -> 180-183, 104 -> 190-193). That is the whole point of the automatic
calibration: the operator arranges the markers in a fixed order instead of typing four ids
per camera into a prompt. This file keeps that arrangement.

What it must NOT do is state which id lands on which *corner*. It used to -- 180 assumed to
be top_left, 181 top_right, and so on, i.e. clockwise from the top-left. The markers are in
fact laid clockwise from the bottom-left, so every corner label came out rotated by 180
degrees relative to reality. Read in CORNER_ORDER the resulting quad winds anticlockwise,
`camera_stitching.calculate_perspective_transform` folds that reflection into the
homography, and the stitched image comes out mirrored -- which ArUco cannot decode at all,
because its codes are rotation-invariant but not mirror-invariant.

So the rig declares the id block per camera (stable, and the operator's own arrangement)
and `assign_corners_by_geometry` recovers the corner roles from where the markers are
actually seen. That keeps the calibration automatic and correct for any starting corner or
camera mounting, without anyone having to re-derive the ordering by hand.
"""

from typing import Dict, Final, Sequence, Tuple

from calibration_contract import CORNER_ORDER, assert_not_mirrored

TABLE_WIDTH_CM: Final = 80.0
TABLE_HEIGHT_CM: Final = 80.0
MARKER_CENTER_OFFSET_CM: Final = 3.0

# Camera IDs intentionally use the last three digits exposed by camera.poll_frame_data.
# Position is the camera's place in the stitching grid -- a mounting fact, not a marker fact.
RIG_CAMERAS: Final = {
    "863": {"position": "top_left"},
    "104": {"position": "top_right"},
}

#: Every id reserved for rig (camera-stitching) calibration, as a flat pool. Deliberately
#: not split per camera and not mapped to corners: both of those have changed with the
#: physical layout before (the tables were re-laid from an interleaved arrangement to one
#: sequential block per table) and hardcoding either is what produced a mirrored
#: calibration. Each camera claims whichever four of these it can actually see, and
#: `assign_corners_by_geometry` decides the corners. Any arrangement, any starting corner,
#: any mounting -- no code change.
RIG_MARKER_IDS: Final[frozenset] = frozenset(range(180, 184)) | frozenset(range(190, 194))

#: How many rig markers each camera must see before its corners can be assigned.
MARKERS_PER_CAMERA: Final = 4


def marker_physical_positions() -> Dict[str, list]:
    """Return marker-center coordinates in table-local centimeters."""
    return {
        "top_left": [MARKER_CENTER_OFFSET_CM, MARKER_CENTER_OFFSET_CM],
        "top_right": [
            TABLE_WIDTH_CM - MARKER_CENTER_OFFSET_CM,
            MARKER_CENTER_OFFSET_CM,
        ],
        "bottom_right": [
            TABLE_WIDTH_CM - MARKER_CENTER_OFFSET_CM,
            TABLE_HEIGHT_CM - MARKER_CENTER_OFFSET_CM,
        ],
        "bottom_left": [
            MARKER_CENTER_OFFSET_CM,
            TABLE_HEIGHT_CM - MARKER_CENTER_OFFSET_CM,
        ],
    }


def assign_corners_by_geometry(
    observed: Dict[int, Sequence[float]], *, context: str = "rig calibration"
) -> Dict[str, int]:
    """Map each observed marker to a table corner using only where it was seen.

    `observed` is `{marker_id: (pixel_x, pixel_y)}` for exactly the four markers of one
    camera. Corner roles come out of the geometry: split the four points into the two
    topmost and the two bottommost by image y (y grows downward), then within each pair the
    smaller x is the left one. This is deterministic for any physical arrangement and does
    not depend on dict iteration order, detection order, or id ordering.

    Raises `ValueError` if the resulting quad is wound in mirrored order -- which cannot
    happen for a genuine four-corner layout, so it means the camera saw something other
    than its four rig markers.
    """
    if len(observed) != 4:
        raise ValueError(
            f"{context}: need exactly 4 markers to assign corners, got "
            f"{sorted(observed)}"
        )

    by_vertical = sorted(observed.items(), key=lambda item: (item[1][1], item[1][0]))
    top_pair, bottom_pair = by_vertical[:2], by_vertical[2:]
    top_left, top_right = sorted(top_pair, key=lambda item: item[1][0])
    bottom_left, bottom_right = sorted(bottom_pair, key=lambda item: item[1][0])

    corners = {
        "top_left": top_left[0],
        "top_right": top_right[0],
        "bottom_right": bottom_right[0],
        "bottom_left": bottom_left[0],
    }
    assert_not_mirrored(
        [observed[corners[corner]] for corner in CORNER_ORDER], context=context
    )
    return corners


def build_fixed_camera_setup() -> dict:
    """Build the legacy calibration structure, with corners still to be measured.

    Each corner carries its physical position (a property of the table, known up front) but
    no `id` and no `pixel_position`: those are filled in by `automatic_calibration` once it
    has actually seen the markers.
    """
    physical_positions = marker_physical_positions()
    setup = {}
    for camera_id, camera in RIG_CAMERAS.items():
        setup[camera_id] = {
            "position": camera["position"],
            "calibration_markers": {
                corner: {"physical_position": physical_positions[corner].copy()}
                for corner in CORNER_ORDER
            },
            "measurements": {
                "width": TABLE_WIDTH_CM,
                "height": TABLE_HEIGHT_CM,
                "marker_offset": MARKER_CENTER_OFFSET_CM,
            },
        }
    return setup
