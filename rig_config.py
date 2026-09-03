"""Static physical configuration for the two-table camera rig.

What this file must NOT do: state which marker id sits at which corner. It used to, as a
tidy per-camera block (863 -> 180/181/182/183 clockwise, 104 -> 190/191/192/193), and that
guess was wrong -- the rig's real, measured layout is interleaved. The legacy flow never
guessed: `save_calibration_markers._prompt_marker_id` asked the operator, corner by corner,
which id was physically there. Replacing operator knowledge with an assumed ordering
produced a calibration whose corner labels were vertically flipped, hence a mirrored
perspective transform, hence a stitched image ArUco could not decode at all.

So the rig declares only which marker ids belong to which camera -- a fact that is stable
and cheap to state -- and `assign_corners_by_geometry` recovers the corner roles from where
the markers are actually seen. That keeps the calibration automatic (no manual id entry)
without inventing a layout.
"""

from typing import Dict, Final, Sequence, Tuple

from calibration_contract import CORNER_ORDER, assert_not_mirrored

TABLE_WIDTH_CM: Final = 80.0
TABLE_HEIGHT_CM: Final = 80.0
MARKER_CENTER_OFFSET_CM: Final = 3.0

# Camera IDs intentionally use the last three digits exposed by camera.poll_frame_data.
# `marker_ids` is a SET of the four ids each camera can see -- deliberately not a
# corner->id mapping. Which of them lands at which corner is measured, never assumed.
RIG_CAMERAS: Final = {
    "863": {
        "position": "top_left",
        "marker_ids": (180, 182, 192, 193),
    },
    "104": {
        "position": "top_right",
        "marker_ids": (181, 183, 190, 191),
    },
}


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
