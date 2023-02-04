import cv2

from src.FaceDetection import FaceDetection
from src.Recognizer import Recognizer


class FaceDetector(Recognizer):
    def __init__(self):
        super().__init__()
        self.debugging = False

    def load_model(self, model_path='../output/model.xml'):
        # load the model from the given path
        self.recognizer.read(model_path)

    def detect_faces(self, frame):
        return self.face_cascade.detectMultiScale(frame, 1.3, 5)

    def recognize(self, frame, mark_face=True):
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
        # draw the label and confidence on the frame
        cv2.rectangle(frame, (detection.x, detection.y), (detection.x + detection.w, detection.y + detection.h),
                      (0, 255, 0), 2)
        label = self.get_person_name(detection.label)
        cv2.putText(frame, "{}, {}%".format(label, round(detection.confidence)),
                    (detection.x, detection.y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
