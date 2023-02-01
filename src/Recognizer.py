import cv2


class Recognizer:
    def __init__(self):
        # Load the Haar cascades
        self.face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')
        # Initialize the recognizer
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
