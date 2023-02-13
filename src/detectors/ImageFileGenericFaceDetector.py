"""
It's a face detector that uses the OpenCV library to detect faces in images
"""
import os

import cv2

from src import CASCADE_DIR, DEFAULT_IMAGES_PATH
from src.FaceDetection import FaceDetection
from src.detectors.ImageFileFaceDetector import ImageFileFaceDetector


class ImageFileGenericFaceDetector(ImageFileFaceDetector):
    def run(self, path):
        """
        It takes a path to an image folder, and returns a score based on how many frames contain a face
        
        :param path: The path to the images
        :return: The score of the model.
        """
        score = 0

        for frame in self.get_images(path):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_detections = self.detect_faces(gray)
            if len(face_detections):
                score += 1
            else:
                score -= 1

            for (x, y, w, h) in face_detections:
                detection = FaceDetection(x, y, w, h)
                self.rectangle_around_face(frame, detection)

            if self.debugging:
                print(face_detections)
                self.show_frame(frame)

        return score


if __name__ == "__main__":
    default_cascade_classifier = os.path.join(CASCADE_DIR, 'haarcascade_frontalface_default.xml')
    input_images_path = input(f"path ({DEFAULT_IMAGES_PATH}): ") or DEFAULT_IMAGES_PATH
    input_cascade_classifier = input(f"Cascade ({default_cascade_classifier}): ") or default_cascade_classifier

    detector = ImageFileGenericFaceDetector(cascade_classifier=input_cascade_classifier, debugging=True)
    detector.run(path=input_images_path)
