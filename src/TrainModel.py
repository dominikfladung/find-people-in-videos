import os
import cv2
import numpy as np
import time
from src.Recognizer import Recognizer


class ModelTrainer(Recognizer):
    def train(self):
        faces, labels = self.prepare_dataset("traindata")
        self.recognizer.train(faces, np.array(labels))

    def save(self, path='../output/model.xml'):
        self.recognizer.write(path)

    # Function to detect the face and return the coordinates
    def detect_face(self, img):
        face_img = img.copy()
        face_rects = self.face_cascade.detectMultiScale(face_img)

        for (x, y, w, h) in face_rects:
            cv2.rectangle(face_img, (x, y), (x + w, y + h), (255, 255, 255), 10)

        return face_img, face_rects

    # Function to prepare the dataset
    def prepare_dataset(self, data_folder_path):
        # Get the directories
        dirs = os.listdir(data_folder_path)

        faces = []
        labels = []
        people_counter = 0
        people_register = dict()

        # Let's go through each directory and read images within it
        for dir_name in dirs:
            people_counter += 1
            people_register[people_counter] = dir_name
            dir_path = data_folder_path + "/" + dir_name
            self.prepare_dataset_folder(dir_path, faces, labels, people_counter)

        cv2.destroyAllWindows()
        cv2.waitKey(1)
        cv2.destroyAllWindows()

        self.set_people_register(people_register)

        return faces, labels

    def prepare_dataset_folder(self, dir_path, faces, labels, people_counter):
        # Get the images names that are inside the given subject directory
        is_dir = os.path.isdir(dir_path)
        if not is_dir:
            return

        images_names = os.listdir(dir_path)

        # go through each image name and read image
        for image_name in images_names:
            image_path = dir_path + "/" + image_name
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            face_img, rect = self.detect_face(gray)

            faces.append(face_img)
            labels.append(people_counter)


if __name__ == "__main__":
    start_time = time.time_ns()
    trainer = ModelTrainer()
    trainer.train()
    trainer.save()
    duration = start_time - time.time_ns()
    print("Done in", duration)
