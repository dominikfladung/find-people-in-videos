"""
The Recognizer class is a base class that contains the methods for handling people register and base methods for face detection
"""
import cv2

from src.FaceRecognition import FaceRecognition
from src.PeopleRegisterManager import PeopleRegisterManager


class FaceRecognizer:
    def __init__(self, cascade_classifier='../cascades/data/haarcascade_frontalface_default.xml', debugging=False):
        self.face_cascade = cv2.CascadeClassifier(cascade_classifier)
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.people_register_manager = PeopleRegisterManager()
        self.debugging = debugging

    def load_model(self, model_path='../output/model.xml'):
        """
        The function reads the model from the given path and loads it into the recognizer.

        :param model_path: The path to the model file, defaults to ../output/model.xml (optional)
        """
        # load the model from the given path
        self.recognizer.read(model_path)

    def recognize(self, image, mark_face=True):
        """
        It takes an image, converts it to grayscale, detects faces, and then for each face, it predicts
        the label and confidence of the face

        :param image: The image to recognize faces in
        :param mark_face: If True, the face will be marked with a rectangle and the label of the face,
        defaults to True (optional)
        :return: A list of FaceDetection objects.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detect_faces(gray)
        face_detections = []

        for (x, y, w, h) in faces:
            # predict the label of the face
            label, confidence = self.recognizer.predict(gray[y:y + h, x:x + w])
            detection = FaceRecognition(label, confidence, x, y, w, h)
            face_detections.append(detection)

            if mark_face:
                self.mark_face(image, detection)

        return face_detections

    def detect_faces(self, image):
        """
        It takes an image as input, and returns a list of rectangles where it thinks it found a face

        :param image: The image to detect faces in
        :return: The detectMultiScale function is a general function that detects objects. Since we are
        calling it on the face cascade, that’s what it detects. The first option is the grayscale image.
        The second is the scaleFactor. Since some faces may be closer to the camera, they would appear
        bigger than the faces in the back. The scale factor compensates for this. The detection
        algorithm
        """
        return self.face_cascade.detectMultiScale(image, 1.3, 5)

    def mark_face(self, image, detection):
        """
        It takes an image and a detection object as input, and draws a rectangle around the face and
        writes the name of the person and the confidence level on the image

        :param image: The image to draw the detection on
        :param detection: The detection object returned by the detector
        """
        # draw the label and confidence on the image
        cv2.rectangle(image, (detection.x, detection.y), (detection.x + detection.w, detection.y + detection.h),
                      (0, 255, 0), 2)
        label = self.people_register_manager.get_person_name(detection.label)
        cv2.putText(image, f"{label}, {round(detection.confidence)}%", (detection.x, detection.y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
