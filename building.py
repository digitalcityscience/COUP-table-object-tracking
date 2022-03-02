from typing import List


class Building:
    def __init__(self, id:int, pos:List[float], lastSeen:int):
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