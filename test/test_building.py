from distutils.command.build import build
import time
from building import Building, Buildings, printJSON


def test_printJSON():
    buildingsDict = {
        19: Building(19, (449, 614, 78.90624111411), time.time(), "cameraId1")
    }
    assert printJSON(buildingsDict) == {19: [449, 614, 78.90624111411, "cameraId1"]}


def test_toJSON():
    building = Building(19, (449, 614, 78.90624111411), time.time(), "cameraId1")
    assert building.toJSON() == '{"19": [449, 614, 78.90624111411, "cameraId1"]}'


def test_pruneUncertainties():
    buildings = [
        Building(19, (449, 614, 78.90624111411), 12121, "cameraId1"),
        Building(19, (450, 614, 78.90624111411), 12121, "cameraId1"),
        Building(20, (450, 614, 78.90624111411), 12121, "cameraId1"),
        Building(19, (450, 614, 78.90624111411), 12121, "cameraId1"),
    ]
    buildings_holder = Buildings()
    buildings_holder.addBuildings(buildings)
    assert buildings_holder.pruneUncertainties() == {
        19: Building(19, (450, 614, 78.90624111411), 12121, "cameraId1",confidence=2)
    }
    

def test_buildings_confidence():
    buildings = [
        Building(19, (449, 614, 78.90624111411), time.time(), "cameraId1"),
        Building(19, (450, 614, 78.90624111411), time.time(), "cameraId1"),
        Building(20, (450, 614, 78.90624111411), time.time(), "cameraId1")
    ]
    buildings_holder = Buildings()
    [buildings_holder.addBuilding(building) for building in buildings]
    assert buildings_holder.toJSON() == '{"19": [450, 614, 78.90624111411, "cameraId1"]}'

