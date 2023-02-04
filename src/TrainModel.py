"""
`ModelTrainer` is a subclass of `Recognizer` that can be used to train a model
"""
import os
import cv2
import numpy as np
import time
from src.Recognizer import Recognizer
from progress.bar import Bar


class ModelTrainer(Recognizer):
    def train(self):
        """
        It takes the images and labels from the prepare_dataset function and trains the recognizer with
        them
        """

        faces, labels = self.prepare_dataset("traindata")
        print("Start Training")
        self.recognizer.train(faces, np.array(labels))

    def save(self, path='../output/model.xml'):
        """
        It saves the model to a file

        :param path: The path to the file where the model will be saved, defaults to ../output/model.xml
        (optional)
        """
        self.recognizer.write(path)

    # Function to detect the face and return the coordinates
    def detect_face(self, img):
        """
        It takes an image as input and returns an image with rectangles around the faces and the
        coordinates of the rectangles

        :param img: The image in which we want to detect faces
        :return: The face_img is the image with the rectangle drawn on it. The face_rects is the
        coordinates of the rectangle.
        """
        face_img = img.copy()
        face_rects = self.face_cascade.detectMultiScale(face_img)

        for (x, y, w, h) in face_rects:
            cv2.rectangle(face_img, (x, y), (x + w, y + h),
                          (255, 255, 255), 10)

        return face_img, face_rects

    def prepare_dataset(self, data_folder_path):
        """
        It takes a folder path as input, and returns two lists: one containing the images, and the other
        containing the labels

        :param data_folder_path: The path to the folder that contains the images of the people you want
        to recognize
        :return: faces and labels
        """
        # Get the directories
        dirs = os.listdir(data_folder_path)

        faces = []
        labels = []
        people_counter = 0
        people_register = dict()

        print("Start preparing dataset")

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

        print("")
        print("Done preparing dataset")

        return faces, labels

    def prepare_dataset_folder(self, dir_path, faces, labels, people_counter):
        """
        It takes a directory path, a list of faces, a list of labels, and a people counter. It then gets
        the images names that are inside the given subject directory, goes through each image name and
        reads the image, and then appends the face image and the label to the lists.

        :param dir_path: The path to the directory that contains the images of the person we want to
        train the model on
        :param faces: A list of face images
        :param labels: A list of labels, i.e. the names of the people on the images
        :param people_counter: The number of people in the dataset
        :return: The face_img and rect are being returned.
        """
        # Get the images names that are inside the given subject directory
        is_dir = os.path.isdir(dir_path)
        if not is_dir:
            return

        images_names = os.listdir(dir_path)
        with Bar('Running: ' + dir_path, suffix='%(percent).1f%%', max=len(images_names)) as bar:
            # go through each image name and read image
            for i, image_name in enumerate(images_names):
                image_path = dir_path + "/" + image_name
                image = cv2.imread(image_path)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                face_img, rect = self.detect_face(gray)

                faces.append(face_img)
                labels.append(people_counter)
                bar.next()


if __name__ == "__main__":
    start_time = time.time()
    trainer = ModelTrainer()
    trainer.train()
    trainer.save()
    duration = time.time() - start_time
    print("Done in", str(duration) + "s")
