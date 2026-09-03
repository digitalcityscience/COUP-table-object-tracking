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
            Marker(201, (30, 40, 0.0), time.time(), "000"),
        ]
    )

    assert markers_holder.toDict() == {
        200: [10, 20, 0.0, "000"],
        201: [30, 40, 0.0, "000"],
    }


def test_calibration_markers_do_not_silence_the_building_feed():
    """A visible calibration marker must not blank out the buildings.

    `toDict` used to return *only* the calibration markers as soon as one of them was
    visible. That branch was dead while `calibrationMarkerIds` held 100-103 (ids the
    frontend never projects), so buildings always flowed. Correcting the list to the
    200-203 the frontend really shows woke the branch up, and any calibration marker left
    on the table then silenced every building for as long as it stayed there.
    """
    markers_holder = Markers()
    markers_holder.clear()
    markers_holder.addMarkers(
        [
            # A building seen twice in one window: passes the confidence gate.
            Marker(12, (50, 60, 0.0), 12121, "000"),
            Marker(12, (51, 61, 0.0), 12121, "000"),
            # A calibration marker seen once: exempt from the gate.
            Marker(200, (10, 20, 0.0), 12121, "000"),
        ]
    )

    snapshot = markers_holder.toDict()
    assert snapshot == {
        12: [51, 61, 0.0, "000"],
        200: [10, 20, 0.0, "000"],
    }


def test_building_markers_still_need_confidence():
    """The calibration exemption must not leak into the building path."""
    markers_holder = Markers()
    markers_holder.clear()
    markers_holder.addMarker(Marker(12, (50, 60, 0.0), 12121, "000"))

    assert markers_holder.toDict() == {}


def test_marker_ids_stay_numbers_in_the_snapshot():
    """The frontend does Number(key) on the JSON object keys; ids must be numeric here."""
    markers_holder = Markers()
    markers_holder.clear()
    markers_holder.addMarker(Marker(203, (1350, 150, 0.0), 12121, "000"))

    snapshot = markers_holder.toDict()
    assert list(snapshot) == [203]
    assert all(isinstance(key, int) for key in snapshot)


def test_calibration_markers_survive_a_single_sighting():
    """One stitched frame is enough for a calibration marker to reach the frontend.

    `server._detection_worker` clears the holder every 200 ms, so a confidence gate here
    would mean "seen twice within one 200 ms window" -- unreachable below ~10 fps, which is
    routine when two camera streams are being stitched. The frontend runs its own stability
    check across snapshots before accepting a reading, so gating again here only ever
    silences calibration entirely.
    """
    markers_holder = Markers()
    markers_holder.clear()
    markers_holder.addMarker(Marker(202, (70, 80, 0.0), time.time(), "000"))

    assert markers_holder.toDict() == {202: [70, 80, 0.0, "000"]}
