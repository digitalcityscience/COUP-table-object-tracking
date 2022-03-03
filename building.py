from typing import Dict, List, Union

from detection import Corner, normalizeCorners


class Building:
    def __init__(self, id: int, pos: List[Union[float, int]], lastSeen: int):
        self.id = id
        self.pos = pos
        self.confidence = 0
        self.lastSeen = lastSeen

    def updateConfidence(self, currentLoop):
        self.confidence = currentLoop - self.lastSeen

    def updatePosition(self, pos, loopcount):
        self.pos = pos
        self.lastSeen = loopcount

    def getConfidence(self):
        return self.confidence

    def getPos(self):
        return self.pos

    def getID(self):
        return self.id


def add_detected_buildings_to_dict(
    ids: List[int],
    corners: List[Corner],
    loopcount: int,
    buildingDict: Dict[int, Building],
) -> None:
    if ids is not None:
        for i in range(0, len(ids)):
            markerID = int(ids[i])

            if markerID is not 500:
                pos = normalizeCorners(corners[i])

                if markerID not in buildingDict:
                    buildingDict[markerID] = Building(int(ids[i]), pos, loopcount)
                else:
                    buildingDict[markerID].updatePosition(pos, loopcount)


def discard_low_confidence_buildings(
    buildingDict: Dict[int, Building], loopcount: int
) -> None:
    for x in list(buildingDict):
        buildingDict[x].updateConfidence(loopcount)
        if buildingDict[x].getConfidence() > 5:  # if not found after 5 loops, discard
            buildingDict.pop(x)


def printJSON(buildingDict: Dict[int, Building]) -> Dict[int, List[float]]:
    jsonDict = {}
    parentDict = {}

    for i in buildingDict:
        jsonDict[i] = buildingDict[i].getPos()

    parentDict["table_state"] = jsonDict
    return jsonDict
