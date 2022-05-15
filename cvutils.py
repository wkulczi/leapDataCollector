import numpy as np
from operator import itemgetter
import cv2


def getImageFixedHeight(oldImg, newHeight, newWidth, fillVal=0, isGrayscale=True):
    if isGrayscale:
        oldHeight, oldWidth = oldImg.shape
        result = np.full((newHeight, newWidth), fillVal, dtype=np.uint8)
    else:
        oldHeight, oldWidth, oldDepth = oldImg.shape
        result = np.full((newHeight, newWidth, oldDepth), fillVal, dtype=np.uint8)

    xCenter = (newWidth - oldWidth) // 2
    yCenter = (newHeight - oldHeight) // 2

    result[yCenter: yCenter + oldHeight, xCenter:xCenter + oldWidth] = oldImg

    return result


def getImageBorders(coordsList):
    topY = min(coordsList, key=itemgetter(1))[1]
    bottomY = max(coordsList, key=itemgetter(1))[1]
    startX = min(coordsList, key=itemgetter(0))[0]
    endX = max(coordsList, key=itemgetter(0))[0]
    return topY, bottomY, startX, endX


# todo rename to drawPointOnCanvas
def drawJointPosOnCanvas(jointCanvas, colorCoords):
    for entry in colorCoords:
        jointCanvas = cv2.circle(jointCanvas, (entry["coords"]),
                                 radius=2, color=entry["color"], thickness=-2)
    return jointCanvas
