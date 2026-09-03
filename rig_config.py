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

from calibration_contract import CORNER_ORDER, assert_not_mirrored, signed_area

TABLE_WIDTH_CM: Final = 80.0
TABLE_HEIGHT_CM: Final = 80.0
MARKER_CENTER_OFFSET_CM: Final = 3.0

# Camera IDs intentionally use the last three digits exposed by camera.poll_frame_data.
# `position` is the camera's place in the stitching grid -- a mounting fact.
# `marker_ids` is the sequential block belonging to that table: the rig was deliberately
# laid out this way so nobody has to type four ids per camera into a prompt.
#
# This must stay a per-camera SET, not a shared pool. The cameras' fields of view overlap,
# so camera 104 can see 863's markers: on the live rig it read 183 near the centre of its
# own frame. Accepting "whichever four rig markers turn up" therefore silently swapped a
# missing 192 for a foreign 183 and produced a nonsense quad. Which of a table's own four
# ids lands on which corner is still never assumed -- that comes from the geometry.
RIG_CAMERAS: Final = {
    "863": {
        "position": "top_left",
        "marker_ids": (180, 181, 182, 183),
    },
    "104": {
        "position": "top_right",
        "marker_ids": (190, 191, 192, 193),
    },
}

#: Every id reserved for rig (camera-stitching) calibration, across both tables.
RIG_MARKER_IDS: Final[frozenset] = frozenset(
    marker_id
    for camera in RIG_CAMERAS.values()
    for marker_id in camera["marker_ids"]
)

#: How many rig markers each camera must see before its corners can be assigned.
MARKERS_PER_CAMERA: Final = 4

#: Minimum fraction of its bounding box that the four markers' quad must cover before it is
#: accepted as a table outline. See `assign_corners_by_geometry` for the measured values.
MIN_QUAD_FILL_RATIO: Final = 0.75


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

    # Four table corners fill their own bounding box almost completely, even under a fair
    # amount of perspective keystone. A stray marker somewhere in the middle of the frame
    # collapses the quad into a dart while leaving the bounding box unchanged, so the fill
    # ratio is what separates them. Measured on the live rig: a correct camera-863 read
    # fills 0.965 and a correct camera-104 read 0.958, while the bad camera-104 read that
    # picked up a foreign marker at frame centre filled only 0.443.
    quad = [
        top_left[1][:2],
        top_right[1][:2],
        bottom_right[1][:2],
        bottom_left[1][:2],
    ]
    xs = [point[0] for point in quad]
    ys = [point[1] for point in quad]
    bounding_area = (max(xs) - min(xs)) * (max(ys) - min(ys))
    fill = abs(signed_area(quad)) / bounding_area if bounding_area else 0.0
    if fill < MIN_QUAD_FILL_RATIO:
        raise ValueError(
            f"{context}: markers {sorted(observed)} do not outline a table -- they fill "
            f"only {fill:.2f} of their bounding box (a real four-corner read fills about "
            f"0.96, minimum accepted {MIN_QUAD_FILL_RATIO}). Positions were "
            f"{ {marker_id: [round(c, 1) for c in point[:2]] for marker_id, point in sorted(observed.items())} }. "
            "Most likely one of this camera's own markers was occluded and a marker from "
            "the neighbouring table was picked up in its place."
        )

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
