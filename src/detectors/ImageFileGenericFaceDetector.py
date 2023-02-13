"""
It's a face detector that uses the OpenCV library to detect faces in images
"""
import os

import cv2

from src import CASCADE_DIR, DEFAULT_IMAGES_PATH
from src.FaceDetection import FaceDetection
from src.FaceRecognizer import FaceRecognizer
import os
import cv2


class ImageFileGenericFaceDetector(FaceRecognizer):
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

    @staticmethod
    def show_frame(frame):
        """
        It takes a frame, resizes it to 800x800, and displays it
        
        :param frame: The frame to be displayed
        """
        ratio = frame.shape[1] / frame.shape[0]
        height = 800
        width = round(height * ratio)
        smaller_frame = cv2.resize(frame, (width, height))
        cv2.imshow("Frame", smaller_frame)
        cv2.waitKey()

    @staticmethod
    def get_images(path):
        """
        It takes a path to a directory, reads all the images in that directory, and returns a list of
        images

        :param path: The path to the folder containing the images
        :return: A list of images
        """
        files = os.listdir(path)
        images = []
        for file in files:
            images.append(cv2.imread(path + "/" + file))

        return images


if __name__ == "__main__":
    default_cascade_classifier = os.path.join(CASCADE_DIR, 'haarcascade_frontalface_default.xml')
    input_images_path = input(f"path ({DEFAULT_IMAGES_PATH}): ") or DEFAULT_IMAGES_PATH
    input_cascade_classifier = input(f"Cascade ({default_cascade_classifier}): ") or default_cascade_classifier

    detector = ImageFileGenericFaceDetector(cascade_classifier=input_cascade_classifier, debugging=True)
    detector.run(path=input_images_path)
