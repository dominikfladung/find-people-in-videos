"""
The Recognizer class is a base class that contains the methods for handling people register and base methods for face detection
"""
import cv2
import json
import os


class Recognizer:
    def __init__(self):
        self.register_json_path = '../output/people_register.json'
        # Load the Haar cascades
        self.face_cascade = cv2.CascadeClassifier(
            'cascades/data/haarcascade_frontalface_default.xml')
        # Initialize the recognizer
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.people_register = self.get_people_register()

    def get_people_register(self):
        """
        If the file exists, open it and return the contents
        :return: A dictionary of the people in the register.
        """
        if not os.path.isfile(self.register_json_path):
            return

        with open(self.register_json_path) as json_file:
            return json.load(json_file)

    def set_people_register(self, dictionary):
        """
        It takes a dictionary as an argument and writes it to a json file

        :param dictionary: The dictionary that you want to save to the json file
        """
        with open(self.register_json_path, 'w') as f:
            json.dump(dictionary, f)

    def get_person_index(self, name):
        """
        It returns the index of the person in the list of people, or -1 if the person is not in the list

        :param name: The name of the person to be added
        :return: The index of the person in the list.
        """
        for i, k in enumerate(self.people_register):
            if k == name:
                return i
        return -1

    def get_person_name(self, index):
        """
        It returns the name of the person at the given index in the people register

        :param index: The index of the person in the people_register dictionary
        :return: The name of the person at the given index.
        """
        return self.people_register[str(index)]
