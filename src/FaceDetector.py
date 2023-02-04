import cv2

from src.FaceDetection import FaceDetection
from src.Recognizer import Recognizer


# > This class is a subclass of the Recognizer class, and it has a method called detect_faces that
# takes in an image and returns a list of bounding boxes for each face detected in the image
class FaceDetector(Recognizer):
    def __init__(self):
        super().__init__()
        self.debugging = False

    def load_model(self, model_path='../output/model.xml'):
        """
        The function reads the model from the given path and loads it into the recognizer.
        
        :param model_path: The path to the model file, defaults to ../output/model.xml (optional)
        """
        # load the model from the given path
        self.recognizer.read(model_path)

    def detect_faces(self, frame):
        """
        It takes a frame as input, and returns a list of rectangles where it thinks it found a face
        
        :param frame: The frame to detect faces in
        :return: The detectMultiScale function is a general function that detects objects. Since we are
        calling it on the face cascade, that’s what it detects. The first option is the grayscale image.
        The second is the scaleFactor. Since some faces may be closer to the camera, they would appear
        bigger than the faces in the back. The scale factor compensates for this. The detection
        algorithm
        """
        return self.face_cascade.detectMultiScale(frame, 1.3, 5)

    def recognize(self, frame, mark_face=True):
        """
        It takes a frame, converts it to grayscale, detects faces, and then for each face, it predicts
        the label and confidence of the face
        
        :param frame: The frame to recognize faces in
        :param mark_face: If True, the face will be marked with a rectangle and the label of the face,
        defaults to True (optional)
        :return: A list of FaceDetection objects.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detect_faces(gray)

        face_detections = []

        for (x, y, w, h) in faces:
            # predict the label of the face
            label, confidence = self.recognizer.predict(gray[y:y + h, x:x + w])
            detection = FaceDetection(label, confidence, x, y, w, h)
            face_detections.append(detection)

            if mark_face:
                self.mark_face(frame, detection)

        return face_detections

    def mark_face(self, frame, detection):
        """
        It takes a frame and a detection object as input, and draws a rectangle around the face and
        writes the name of the person and the confidence level on the frame
        
        :param frame: The frame to draw the detection on
        :param detection: The detection object returned by the detector
        """
        # draw the label and confidence on the frame
        cv2.rectangle(frame, (detection.x, detection.y), (detection.x + detection.w, detection.y + detection.h),
                      (0, 255, 0), 2)
        label = self.get_person_name(detection.label)
        cv2.putText(frame, "{}, {}%".format(label, round(detection.confidence)),
                    (detection.x, detection.y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
