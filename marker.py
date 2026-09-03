import json
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

from calibration_contract import MAP_CALIBRATION_MARKER_IDS
from detection import Corner, normalizeCorners

Position = Tuple[int, int, float]
CameraId = Union[int, str]
# Sorted list view of the one contract in `calibration_contract`; never redeclared here.
calibrationMarkerIds = sorted(MAP_CALIBRATION_MARKER_IDS)

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
    # Per-instance, not shared. This was a bare class attribute, so every Markers() aliased
    # one dict: two holders silently accumulated into each other, and the confidence counts
    # one of them reported belonged partly to the other. Production only ever builds one
    # holder (`server._detection_worker`), so this is behaviour-preserving there.
    mDict: Dict[int, Marker]

    def __init__(self):
        self.mDict = {}

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

    def reduceToCalibrationMarkers(self) -> Dict[int, Marker]: 
        result = {}

        for marker in self.mDict.values():
            if marker.id in calibrationMarkerIds:
                # No confidence gate here, unlike pruneUncertainties: the holder is cleared
                # every 200 ms (server._detection_worker), so requiring confidence >= 1 means
                # requiring two stitched frames inside one 200 ms window. Below ~10 fps -- routine
                # when stitching two cameras -- that never happens and the calibration markers are
                # never forwarded at all. The frontend does its own stability check across
                # snapshots before it accepts a reading, so the gate belongs there, not here.
                result[marker.id] = marker

        return result


    def foundCalibrationMarkers(self)-> bool:
        for calibMarkerId in calibrationMarkerIds:
            if calibMarkerId in self.mDict.keys():
                return True

        return False


    def checkConfidence(self, marker: Marker):
        if marker.confidence >= 1:
            return True
        
        return False

    def toJSON(self) -> str:
        return json.dumps(self.toDict())

    def toDict(self) -> Dict[int, List[float]]:
        """One snapshot carrying both marker kinds, each on its own admission rule.

        Building markers keep the confidence gate: a single stray read should not place a
        building on the table. Calibration markers bypass it, because the holder is cleared
        every 200 ms (`server._detection_worker`) and "confidence >= 1" therefore means
        "seen twice inside one 200 ms window" -- unreachable below ~10 fps, which is routine
        when two camera streams are being stitched. The frontend runs its own stability
        check across snapshots before accepting a calibration reading, so the gate belongs
        there, not here.

        Crucially the two sets are merged rather than one replacing the other. This used to
        return *only* the calibration markers as soon as any one of them was visible, which
        silenced the entire building feed for as long as a calibration marker stayed on the
        table. That was harmless while `calibrationMarkerIds` still held 100-103 (ids the
        frontend never projects, so the branch never fired) and became a live regression the
        moment the list was corrected to the 200-203 the frontend actually shows.
        """
        markers = self.pruneUncertainties()
        markers.update(self.reduceToCalibrationMarkers())
        return printJSON(markers)


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
