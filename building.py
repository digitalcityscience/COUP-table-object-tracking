import time
from dataclasses import dataclass
from typing import Dict, List, Union

from detection import Corner, normalizeCorners

Position = List[Union[float, int]]
CameraId = Union[int, str]


@dataclass
class Building:
    id: int
    position: Position
    lastSeen: float
    cameraId: CameraId
    confidence: int = 0

    def updateConfidence(self, currentLoop):
        self.confidence = currentLoop - self.lastSeen

    def updatePosition(self, pos, loopcount):
        self.position = pos
        self.lastSeen = loopcount

    def getConfidence(self):
        return self.confidence

    def getPos(self):
        return self.position

    def getID(self):
        return self.id


BuildingDictionary = Dict[int, Building]


def add_detected_buildings_to_dict(
    ids: List[int],
    cameraId: CameraId,
    corners: List[Corner],
    loopcount: int,
    buildingDict: BuildingDictionary,
) -> None:
    if ids is not None:
        for i in range(0, len(ids)):
            markerID = int(ids[i])

            if markerID is not 500:
                position = normalizeCorners(corners[i])

                if markerID not in buildingDict:
                    buildingDict[markerID] = Building(
                        id=markerID,
                        position=position,
                        lastSeen=loopcount,
                        cameraId=cameraId,
                    )
                else:
                    buildingDict[markerID].updatePosition(position, loopcount)


def map_detected_buildings(
    cameraId: CameraId, ids: List[int], corners: List[Corner]
) -> Dict[int, Building]:
    buildingDict: Dict[int, Building] = {}
    if ids is not None:
        for i in range(0, len(ids)):
            markerID = int(ids[i])
            now = time.time()

            if markerID is not 500:
                position = normalizeCorners(corners[i])
                buildingDict[markerID] = Building(
                    id=markerID,
                    position=position,
                    lastSeen=now,
                    cameraId=cameraId,
                )
    return buildingDict


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
