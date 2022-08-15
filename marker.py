import json
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

from tomlkit import boolean

from detection import Corner, normalizeCorners

Position = Tuple[int, int, float]
CameraId = Union[int, str]
calibrationMarkerIds = [100,101,102,103]

@dataclass
class Marker:
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


class Markers:
    mDict: Dict[int, Marker] = {}

    def clear(self):
        self.mDict.clear()

    def addMarker(self, marker: Marker):
        if self.mDict.get(marker.id) is None:
            self.mDict[marker.id] = marker
        else:
            self.mDict[marker.id] = Marker(
                id=marker.id,
                position=marker.position,
                lastSeen=12121,
                confidence=self.mDict[marker.id].confidence + 1,
                cameraId=marker.cameraId,
            )

    def addMarkers(self, markers: List[Marker]):
        for marker in markers:
            self.addMarker(marker)

    def pruneUncertainties(self) -> Dict[int, Marker]:
        result = {}
        for marker in self.mDict.values():
            if self.checkConfidence(marker):
                result[marker.id] = marker
        return result

    def foundCalibrationMarkers(self)-> bool:
        for calibMarkerId in calibrationMarkerIds:
            if calibMarkerId in self.mDict.keys():
                return True

        return False

    def reduceToCalibrationMarkers(self) -> Dict[int, Marker]: 
        result = {}

        for marker in self.mDict.values():
            if marker.id in calibrationMarkerIds:
                if self.checkConfidence(marker):
                    result[marker.id] = marker

        return result


    def checkConfidence(self, marker: Marker):
        if marker.confidence >= 2:
            return True
        
        return False

    def toJSON(self) -> str:
        if self.foundCalibrationMarkers():
            return json.dumps(self.reduceToCalibrationMarkers())
            
        return json.dumps(printJSON(self.pruneUncertainties()))
        

        
        


MarkerDictionary = Dict[int, Marker]


def add_detected_markers_to_dict(
    ids: List[int],
    cameraId: CameraId,
    corners: List[Corner],
    loopcount: int,
    markerDict: MarkerDictionary,
) -> None:
    if ids is not None:
        for i in range(0, len(ids)):
            markerID = int(ids[i])

            if markerID is not 500:
                position = normalizeCorners(corners[i])

                if markerID not in markerDict:
                    markerDict[markerID] = Marker(
                        id=markerID,
                        position=position,
                        lastSeen=loopcount,
                        cameraId=cameraId,
                    )
                else:
                    markerDict[markerID].updatePosition(position, loopcount)


def map_detected_markers(
    cameraId: CameraId, ids: List[int], corners: List[Corner]
) -> Dict[int, Marker]:
    markerDict: Dict[int, Marker] = {}
    if ids is not None:
        for i in range(0, len(ids)):
            markerID = int(ids[i])
            now = time.time()

            if markerID is not 500:
                position = normalizeCorners(corners[i])
                markerDict[markerID] = Marker(
                    id=markerID,
                    position=position,
                    lastSeen=now,
                    cameraId=cameraId,
                )
    return markerDict


def discard_low_confidence_markers(
    markerDict: Dict[int, Marker], loopcount: int
) -> None:
    for x in list(markerDict):
        markerDict[x].updateConfidence(loopcount)
        if markerDict[x].getConfidence() > 5:  # if not found after 5 loops, discard
            markerDict.pop(x)


def printJSON(markerDict: Dict[int, Marker]) -> Dict[int, List[float]]:
    jsonDict = {}
    parentDict = {}

    for i in markerDict:
        jsonDict[i] = [*(markerDict[i].getPos()), markerDict[i].cameraId]

    parentDict["table_state"] = jsonDict
    return jsonDict
