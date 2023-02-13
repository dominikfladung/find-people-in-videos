"""
It takes a YouTube video URL, downloads the video, and then uses the `VideoFaceDetector` class to
detect faces in the video
"""
import re

from src import DEFAULT_MODEL_PATH
from src.detectors.VideoFaceDetector import VideoFaceRecognizer
from src.utils.YoutubeDownloader import YoutubeDownloader


class YoutubeVideoFaceDetector:
    @staticmethod
    def make_url_filename_safe(url):
        """
        It takes a string and replaces all characters that are not alphanumeric, dashes, underscores,
        periods, or spaces with underscores

        :param url: The URL of the page you want to download
        :return: a string that is the url with all characters that are not alphanumeric, underscore,
        dash, period, or space replaced with an underscore.
        """
        return re.sub('[^\w\-_\. ]', '_', url)

    def run(self, url, model_path):
        """
        It downloads a video from a given URL, and then runs the run() function from the parent class

        :param url: The URL of the YouTube video you want to download
        :param model_path: The path to the model folder
        """
        filename = "videos/" + self.make_url_filename_safe(url) + ".mp4"
        YoutubeDownloader().download(url, filename)
        VideoFaceRecognizer().run(filename, model_path)


if __name__ == "__main__":
    input_url = input("Url: ")
    input_model_path = input(f"model_path ({DEFAULT_MODEL_PATH}): ") or DEFAULT_MODEL_PATH
    YoutubeVideoFaceDetector().run(input_url, model_path=input_model_path)
