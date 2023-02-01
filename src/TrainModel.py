import os
import cv2
import numpy as np

from src.Recognizer import Recognizer


class ModelTrainer(Recognizer):
    def train(self):
        # Prepare the dataset
        faces, labels = self.prepare_dataset("traindata")
        self.recognizer.train(faces, np.array(labels))

    def save(self, path='model.xml'):
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

        # List to hold all subject faces
        faces = []
        # List to hold labels for all subjects
        labels = []
        labelcounter = 0
        # Let's go through each directory and read images within it
        for dir_name in dirs:
            labelcounter += 1
            subject_dir_path = data_folder_path + "/" + dir_name
            # Get the images names that are inside the given subject directory
            subject_images_names = os.listdir(subject_dir_path)

            # go through each image name and read image
            for image_name in subject_images_names:
                image_path = subject_dir_path + "/" + image_name
                image = cv2.imread(image_path)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # Detect the face
                face_img, rect = self.detect_face(gray)

                # Add face to list of faces
                faces.append(face_img)
                # Add label to list of labels
                labels.append(labelcounter)

        cv2.destroyAllWindows()
        cv2.waitKey(1)
        cv2.destroyAllWindows()

        return faces, labels


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train()
    trainer.save()
