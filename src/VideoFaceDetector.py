import cv2
import imageio

from src.FaceDetector import FaceDetector


class VideoFaceDetector(FaceDetector):
    def run(self, filename, model_path="model.xml"):
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
                print( str(i) + ' / ' + str(max_frame_index))

            video_writer.append_data(frame)

            key = cv2.waitKey(1) & 0xFF
            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

        # do a bit of cleanup
        cv2.destroyAllWindows()
        video_writer.close()

    def display_frame(self, frame, i, max_frame_index):
        cv2.imshow("Frame", frame)
        cv2.setWindowTitle('Frame', str(i) + ' / ' + str(max_frame_index))


if __name__ == "__main__":
    VideoFaceDetector().run(filename='videos/got.mp4')
