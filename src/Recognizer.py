import cv2
import json
import os


class Recognizer:
    def __init__(self):
        self.register_json_path = '../output/people_register.json'
        # Load the Haar cascades
        self.face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')
        # Initialize the recognizer
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.people_register = self.get_people_register()

    def get_people_register(self):
        if not os.path.isfile(self.register_json_path):
            return

        with open(self.register_json_path) as json_file:
            return json.load(json_file)

    def set_people_register(self, dictionary):
        with open(self.register_json_path, 'w') as f:
            json.dump(dictionary, f)

    def get_person_index(self, name):
        for i, k in enumerate(self.people_register):
            if k == name:
                return i
        return -1

    def get_person_name(self, index):
        return self.people_register[str(index)]
