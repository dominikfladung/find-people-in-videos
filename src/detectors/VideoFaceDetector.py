"""
This class is a subclass of the FaceDetector class, and it is used to detect faces in a video.
"""
import cv2
import imageio

from src.FaceRecognizer import FaceRecognizer


class VideoFaceRecognizer(FaceRecognizer):
    def run(self, filename, model_path='../output/model.xml'):
        """
        It takes a video file, runs the face recognition algorithm on each frame, and outputs a new
        video file with the face recognition results

        :param filename: The path to the video file you want to process
        :param model_path: The path to the model file, defaults to ../output/model.xml (optional)
        """
        self.load_model(model_path)

        output_filename = filename + 'detected.mp4'
        video = imageio.get_reader(filename,  'ffmpeg')
        meta = video.get_meta_data()
        video_writer = imageio.get_writer(output_filename, fps=meta['fps'])
        max_frame_index = round(meta['duration'] * meta['fps'])

        for i, frame in enumerate(video):
            face_detections = self.recognize(frame)

            if self.debugging:
                print(face_detections)
                self.display_frame(frame, i, max_frame_index)
            else:
                print(str(i) + ' / ' + str(max_frame_index))

            video_writer.append_data(frame)

            key = cv2.waitKey(1) & 0xFF
            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

        # do a bit of cleanup
        cv2.destroyAllWindows()
        video_writer.close()

    @staticmethod
    def display_frame(frame, i, max_frame_index):
        """
        It displays the frame and sets the window title to the current frame number and the total number
        of frames

        :param frame: the frame to display
        :param i: the current frame index
        :param max_frame_index: The total number of frames in the video
        """
        cv2.imshow("Frame", frame)
        cv2.setWindowTitle('Frame', str(i) + ' / ' + str(max_frame_index))


if __name__ == "__main__":
    path = input('Enter the path to the video file: ')
    VideoFaceRecognizer().run(filename=path)
