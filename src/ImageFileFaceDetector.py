import os
import cv2
from src.FaceDetector import FaceDetector


# It's a face detector that uses the OpenCV library to detect faces in images
class ImageFileFaceDetector(FaceDetector):
    def run(self, model_path='../output/model.xml'):
        """
        It loads the model, then for each image in the folder, it detects faces and prints the results,
        then it resizes the image and displays it
        
        :param model_path: The path to the model file, defaults to ../output/model.xml (optional)
        """
        self.load_model(model_path)

        for frame in self.get_images("traindata/kit_harington"):
            face_detections = self.recognize(frame)
            print(face_detections)

            ratio = frame.shape[1] / frame.shape[0]
            height = 800
            width = round(height * ratio)
            smaller_frame = cv2.resize(frame, (width, height))
            cv2.imshow("Frame", smaller_frame)

            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

    def get_images(self, path):
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
    ImageFileFaceDetector().run()
