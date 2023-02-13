"""
It's a face detector that uses the OpenCV library to detect faces in images
"""
from src import DEFAULT_MODEL_PATH, DEFAULT_IMAGES_PATH
from src.detectors.ImageFileGenericFaceDetector import ImageFileGenericFaceDetector


class ImageFileFaceDetector(ImageFileGenericFaceDetector):
    def run(self, model_path=None, path=None):
        """
        It loads the model, then for each image in the folder, it detects faces and prints the results,
        then it resizes the image and displays it

        :param path: the path to the image folder
        :param model_path: The path to the model folder
        """
        self.load_model(model_path)

        for frame in self.get_images(path):
            face_detections = self.recognize(frame)
            print(face_detections)
            self.show_frame(frame)


if __name__ == "__main__":
    input_images_path = input(f"path ({DEFAULT_IMAGES_PATH}): ") or DEFAULT_IMAGES_PATH
    input_model_path = input(f"model_path ({DEFAULT_MODEL_PATH}): ") or DEFAULT_MODEL_PATH
    ImageFileFaceDetector().run(path=input_images_path, model_path=input_model_path)
