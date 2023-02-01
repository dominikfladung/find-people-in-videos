import os
import cv2
from src.FaceDetector import FaceDetector


class ImageFileFaceDetector(FaceDetector):
    def run(self, model_path="model.xml"):
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
        files = os.listdir(path)
        images = []
        for file in files:
            images.append(cv2.imread(path + "/" + file))

        return images


if __name__ == "__main__":
    ImageFileFaceDetector().run()
