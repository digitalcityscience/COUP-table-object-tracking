import json
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

from detection import Corner, normalizeCorners

Position = Tuple[int, int, float]
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

    def toJSON(self) -> str:
        return json.dumps({self.id: [*self.position, self.cameraId]})


class Buildings:
    bDict: Dict[int, Building] = {}

    def clear(self):
        self.bDict.clear()

    def addBuilding(self, building: Building):
        if self.bDict.get(building.id) is None:
            self.bDict[building.id] = building
        else:
            self.bDict[building.id] = Building(
                id=building.id,
                position=building.position,
                lastSeen=12121,
                confidence=self.bDict[building.id].confidence + 1,
                cameraId=building.cameraId,
            )

    def addBuildings(self, buildings: List[Building]):
        for building in buildings:
            self.addBuilding(building)

    def pruneUncertainties(self) -> Dict[int, Building]:
        result = {}
        for building in self.bDict.values():
            if building.confidence >= 1:
                result[building.id] = building
        return result

    def toJSON(self) -> str:
        return json.dumps(printJSON(self.pruneUncertainties()))


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
        jsonDict[i] = [*(buildingDict[i].getPos()), buildingDict[i].cameraId]

    parentDict["table_state"] = jsonDict
    return jsonDict
