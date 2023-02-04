"""
It's a face detector that uses the OpenCV library to detect faces in images
"""
import cv2

from src.FaceDetection import FaceDetection
from src.detectors.ImageFileFaceDetector import ImageFileFaceRecognizer


class ImageFileGenericFaceRecognizer(ImageFileFaceRecognizer):
    def __init__(self, cascade_classifier='../../cascades/data/haarcascade_frontalface_alt2.xml', debugging=False):
        super().__init__(cascade_classifier=cascade_classifier, debugging=debugging)

    def run(self, model_path='../../output/model.xml', path="../traindata/kit_harington"):
        """
        It loads the model, then for each image in the folder, it detects faces and prints the results,
        then it resizes the image and displays it

        :param path: the path to the image folder
        :param model_path: The path to the model file, defaults to ../output/model.xml (optional)
        """
        for frame in self.get_images(path):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_detections = self.detect_faces(gray)

            for (x, y, w, h) in face_detections:
                detection = FaceDetection(x, y, w, h)
                self.rectangle_around_face(frame, detection)

            print(face_detections)
            self.show_frame(frame)


if __name__ == "__main__":
    images_path = input("Path: ")
    ImageFileGenericFaceRecognizer().run(path=images_path)
