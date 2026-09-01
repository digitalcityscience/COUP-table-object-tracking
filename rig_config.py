"""Static physical configuration for the two-table camera rig."""

from typing import Final

TABLE_WIDTH_CM: Final = 80.0
TABLE_HEIGHT_CM: Final = 80.0
MARKER_CENTER_OFFSET_CM: Final = 3.0

CORNER_ORDER: Final = (
    "top_left",
    "top_right",
    "bottom_right",
    "bottom_left",
)

# Camera IDs intentionally use the last three digits exposed by camera.poll_frame_data.
# Marker order is clockwise from the physical top-left corner of each table.
RIG_CAMERAS: Final = {
    "863": {
        "position": "top_left",
        "marker_ids": {
            "top_left": 180,
            "top_right": 181,
            "bottom_right": 182,
            "bottom_left": 183,
        },
    },
    "104": {
        "position": "top_right",
        "marker_ids": {
            "top_left": 190,
            "top_right": 191,
            "bottom_right": 192,
            "bottom_left": 193,
        },
    },
}


def marker_physical_positions() -> dict[str, list[float]]:
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


def build_fixed_camera_setup() -> dict:
    """Build the legacy calibration structure from the fixed rig definition."""
    physical_positions = marker_physical_positions()
    setup = {}
    for camera_id, camera in RIG_CAMERAS.items():
        setup[camera_id] = {
            "position": camera["position"],
            "calibration_markers": {
                corner: {
                    "id": str(camera["marker_ids"][corner]),
                    "physical_position": physical_positions[corner].copy(),
                }
                for corner in CORNER_ORDER
            },
            "measurements": {
                "width": TABLE_WIDTH_CM,
                "height": TABLE_HEIGHT_CM,
                "marker_offset": MARKER_CENTER_OFFSET_CM,
            },
        }
    return setup
