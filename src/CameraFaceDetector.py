# import the necessary packages
import cv2

from src.FaceDetector import FaceDetector


class CameraFaceDetector(FaceDetector):
    def run(self, model_path="model.xml"):
        self.load_model(model_path)

        # initialize the video capture
        cap = cv2.VideoCapture(0)

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
    CameraFaceDetector().run()