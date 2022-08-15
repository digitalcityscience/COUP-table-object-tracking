from distutils.command.build import build
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

