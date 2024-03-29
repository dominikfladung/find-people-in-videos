"""
`ModelTrainer` is a subclass of `Recognizer` that can be used to train a model
"""
import os
import cv2
import numpy as np
import time

from src import OUTPUT_DIR, CASCADE_DIR, TRAINDATA_DIR
from src.FaceRecognizer import FaceRecognizer
from progress.bar import Bar

from src.PeopleRegisterManager import PeopleRegisterManager


class ModelTrainer(FaceRecognizer):
    def __init__(self, cascade_classifier=CASCADE_DIR + '/haarcascade_frontalface_default.xml', debugging=False):
        super().__init__(cascade_classifier, debugging)
        self.people_register = dict()

    def train(self, dataset_path=TRAINDATA_DIR, output_path=OUTPUT_DIR + '/model'):
        """
        It takes the images and labels from the prepare_dataset function and trains the recognizer with
        them
        """
        if not os.path.isdir(output_path):
            os.mkdir(output_path)
        else:
            self.clear_dir(output_path)

        faces, labels = self.prepare_dataset(dataset_path, output_path)

        if len(faces):
            print("Start Training")
            self.recognizer.train(faces, np.array(labels))
            self.save(output_path)
        else:
            print("Empty Training data - nothing todo")
            os.rmdir(output_path)

    def save(self, path=OUTPUT_DIR + '/model'):
        """
        It saves the model to a file

        :param path: The path to the file where the model will be saved, defaults to ../output/model.xml
        (optional)
        """
        self.people_register_manager = PeopleRegisterManager(path)
        self.people_register_manager.set_people_register(self.people_register)
        self.recognizer.write(path + "/model.xml")

    def crop_face(self, image):
        """
        Function to detect the face and crop it
        :param image: The image in which we want to detect faces
        :return: cropped image or None if no face was found
        """
        face_rects = self.detect_faces(image)
        if len(face_rects) == 0:
            return None

        (x, y, w, h) = face_rects[0]  # in dataset only images with one person are used
        return image[y:y + h, x:x + w]

    def prepare_dataset(self, data_folder_path, output_path):
        """
        It takes a folder path as input, and returns two lists: one containing the images, and the other
        containing the labels

        :param output_path: the model path
        :param data_folder_path: The path to the folder that contains the images of the people you want
        to recognize
        :return: faces and labels
        """
        # Get the directories
        dirs = os.listdir(data_folder_path)

        faces = []
        labels = []
        people_counter = 0

        print("Start preparing dataset")

        # Let's go through each directory and read images within it
        for dir_name in dirs:
            people_counter += 1
            self.people_register[people_counter] = dir_name
            dir_path = data_folder_path + "/" + dir_name
            self.prepare_dataset_folder(dir_path, faces, labels, people_counter, output_path)

        cv2.waitKey(1)
        cv2.destroyAllWindows()

        print("")
        print("Done preparing dataset")

        return faces, labels

    def prepare_dataset_folder(self, dir_path, faces, labels, people_counter, output_path):
        """
        It takes a directory path, a list of faces, a list of labels, and a people counter. It then gets
        the images names that are inside the given subject directory, goes through each image name and
        reads the image, and then appends the face image and the label to the lists.

        :param output_path: the output file path for the model
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
            detection_output_path = output_path + "/traindata/"

            if self.debugging:
                label = self.people_register[people_counter]
                detection_output_subject_path = detection_output_path + "/" + label
                os.makedirs(detection_output_subject_path, exist_ok=True)

            # go through each image name and read image
            for i, image_name in enumerate(images_names):
                image_path = dir_path + "/" + image_name
                origin_image = cv2.imread(image_path)
                gray_image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2GRAY)
                face_image = self.crop_face(gray_image)

                if face_image is not None and face_image.any():
                    faces.append(face_image)
                    labels.append(people_counter)

                    #  save the images which are used to train the model
                    if self.debugging:
                        cv2.imwrite(detection_output_subject_path + "/" + image_name, face_image)

                bar.next()

    def clear_dir(self, path):
        """
        It deletes all files in a directory
        
        :param path: The path to the directory you want to clear
        :return: the path of the file.
        """
        if not os.path.isdir(path):
            return

        for f in os.listdir(path):
            current_path = os.path.join(path, f)
            if os.path.isdir(current_path):
                self.clear_dir(current_path)
            else:
                os.remove(current_path)


if __name__ == "__main__":
    start_time = time.time()
    trainer = ModelTrainer(debugging=True)
    trainer.train()
    duration = time.time() - start_time
    print("Done in", str(duration) + "s")
