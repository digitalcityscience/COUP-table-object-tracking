from typing import Dict
import numpy
import cv2

from building import Building


def draw_status_window(buildingDict: Dict[int, Building]) -> None:
    status = numpy.zeros((800, 320, 3), numpy.uint8)
    statusX = 50
    font = cv2.FONT_HERSHEY_SIMPLEX
    for x in buildingDict:
        ctr = buildingDict[x].getPos()[0]
        deg = buildingDict[x].getPos()[1]
        id = buildingDict[x].getID()
        conf = buildingDict[x].getConfidence()

        cv2.putText(status, str(id), (30, statusX), font, 0.8, (255, 255, 255), 1)
        cv2.putText(status, str(ctr), (100, statusX), font, 0.6, (255, 255, 255), 1)
        cv2.putText(
            status,
            str(int(deg * 180 / 3.14)),
            (220, statusX),
            font,
            0.6,
            (255, 255, 255),
            1,
        )
        cv2.putText(status, str(conf), (300, statusX), font, 0.6, (255, 255, 255), 1)
        statusX += 35
    cv2.namedWindow("Status", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Status", status)
