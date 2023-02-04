# > This class is a data structure that holds the information of a face detected by the face detection algorithm.
class FaceDetection:
    def __init__(self, label, confidence, x, y, w, h):
        self.label = label
        self.confidence = confidence
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def __str__(self):
        return "Label: " + str(self.label) + " Confidence: " + str(self.confidence) + " X: " + str(self.x) + " Y: " +\
            str(self.y) + " W: " + str(self.w) + " H: " + str(self.h)

    def __repr__(self):
        return str(self)
