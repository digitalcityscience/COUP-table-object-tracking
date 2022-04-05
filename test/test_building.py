from distutils.command.build import build
import time
from building import Building, printJSON


def test_printJSON():
    buildingsDict = {
        19: Building(19, (449, 614, 78.90624111411), time.time(), "cameraId1")
    }
    assert printJSON(buildingsDict) == {19: [449, 614, 78.90624111411]}


def test_toJSON():
    building = Building(19, (449, 614, 78.90624111411), time.time(), "cameraId1")
    assert building.toJSON() == '{"19": [449, 614, 78.90624111411]}'
