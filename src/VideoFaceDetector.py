# import the necessary packages
import cv2
import imageio

from src.FaceDetector import FaceDetector


class VideoFaceDetector(FaceDetector):
    def run(self, model_path="model.xml"):
        self.load_model(model_path)

        # initialize the video capture
        filename = 'videos/got.mp4'
        video = imageio.get_reader(filename,  'ffmpeg')

        for frame in video:
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


if __name__ == "__main__":
    VideoFaceDetector().run()
