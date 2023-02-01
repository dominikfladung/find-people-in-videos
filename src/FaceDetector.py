import cv2

from src.FaceDetection import FaceDetection
from src.Recognizer import Recognizer


class FaceDetector(Recognizer):
    def load_model(self, model_path='model.xml'):
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
            face_detections.append(FaceDetection(label, confidence, x, y, w, h))

            if mark_face:
                # draw the label and confidence on the frame
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Label: {}, {}%".format(label, round(confidence)), (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return face_detections
