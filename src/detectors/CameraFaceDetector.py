"""
It's a face detector that uses a camera to detect faces
"""

import cv2

from src import DEFAULT_MODEL_PATH
from src.FaceRecognizer import FaceRecognizer


class CameraFaceDetector(FaceRecognizer):
    def run(self, model_path, capture_index=0):
        """
        We load the model, initialize the video capture, loop over the frames from the video stream,
        grab the current frame, detect faces, print the face detections, show the frame, and if the `q`
        key was pressed, break from the loop
        
        :param model_path: The path to the trained model
        """
        self.load_model(model_path)

        # initialize the video capture
        cap = cv2.VideoCapture(capture_index)

        # loop over the frames from the video stream
        while True:
            # grab the current frame
            ret, frame = cap.read()

            face_detections = self.recognize(frame)
            print(face_detections)

            # show the frame
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

        # do a bit of cleanup
        cv2.destroyAllWindows()
        cap.release()


if __name__ == "__main__":
    input_model_path = input(f"model_path ({DEFAULT_MODEL_PATH}): ") or DEFAULT_MODEL_PATH
    input_capture_index = input(f"capture_index (0):") or 0
    CameraFaceDetector().run(model_path=input_model_path, capture_index=input_capture_index)
