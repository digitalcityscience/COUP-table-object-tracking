import time
from building import Building, printJSON

def test_printJSON():
    buildingsDict = {19: Building(19, [449, 614, 78.90624111411], time.time(), "cameraId1")}
    assert printJSON(buildingsDict) == {19: [449, 614, 78.90624111411]}
