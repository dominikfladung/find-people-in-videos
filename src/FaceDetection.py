"""
This class is a data structure that holds the information of a face detected by the face detection algorithm.
"""


class FaceDetection:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def __str__(self):
        return " X: " + str(self.x) + " Y: " +\
            str(self.y) + " W: " + str(self.w) + " H: " + str(self.h)

    def __repr__(self):
        return str(self)
