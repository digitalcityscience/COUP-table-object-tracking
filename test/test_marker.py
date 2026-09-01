import time
from marker import Marker, Markers, printJSON


def test_printJSON():
    markersDict = {
        19: Marker(19, (449, 614, 78.90624111411), time.time(), "cameraId1")
    }
    assert printJSON(markersDict) == {19: [449, 614, 78.90624111411, "cameraId1"]}


def test_toJSON():
    marker = Marker(19, (449, 614, 78.90624111411), time.time(), "cameraId1")
    assert marker.toJSON() == '{"19": [449, 614, 78.90624111411, "cameraId1"]}'


def test_pruneUncertainties():
    markers = [
        Marker(19, (449, 614, 78.90624111411), 12121, "cameraId1"),
        Marker(19, (450, 614, 78.90624111411), 12121, "cameraId1"),
        Marker(20, (450, 614, 78.90624111411), 12121, "cameraId1"),
        Marker(19, (450, 614, 78.90624111411), 12121, "cameraId1"),
    ]
    markers_holder = Markers()
    markers_holder.addMarkers(markers)
    assert markers_holder.pruneUncertainties() == {
        19: Marker(19, (450, 614, 78.90624111411), 12121, "cameraId1",confidence=2)
    }
    

def test_markers_confidence():
    markers = [
        Marker(19, (449, 614, 78.90624111411), time.time(), "cameraId1"),
        Marker(19, (450, 614, 78.90624111411), time.time(), "cameraId1"),
        Marker(20, (450, 614, 78.90624111411), time.time(), "cameraId1")
    ]
    markers_holder = Markers()
    [markers_holder.addMarker(marker) for marker in markers]
    assert markers_holder.toJSON() == '{"19": [450, 614, 78.90624111411, "cameraId1"]}'


def test_map_calibration_snapshot_keeps_frontend_marker_ids():
    """The four map-calibration markers TOSCA-2 projects are 200-203, not 100-103.

    TOSCA-2 renders `public/collab/calibration-markers/4x4_1000-200.svg`..`-203.svg` at the
    table corners and reads those ids straight back out of this snapshot to build its
    `map_calibration` handshake. While this list said 100-103 the ids never lined up, so the
    frontend saw four ordinary markers instead of its calibration set and could never
    calibrate. (The `4x4_1000` filenames are not a second mismatch: OpenCV's 4x4 dictionaries
    are nested, so ids 200-203 are bit-identical under `DICT_4X4_250`.)
    """
    markers_holder = Markers()
    markers_holder.clear()
    markers_holder.addMarkers(
        [
            Marker(200, (10, 20, 0.0), time.time(), "000"),
            Marker(200, (10, 20, 0.0), time.time(), "000"),
            Marker(201, (30, 40, 0.0), time.time(), "000"),
            Marker(201, (30, 40, 0.0), time.time(), "000"),
            Marker(100, (50, 60, 0.0), time.time(), "000"),
            Marker(100, (50, 60, 0.0), time.time(), "000"),
        ]
    )

    assert markers_holder.toDict() == {
        200: [10, 20, 0.0, "000"],
        201: [30, 40, 0.0, "000"],
    }
