import re

from src.VideoFaceDetector import VideoFaceDetector
from src.YoutubeDownloader import YoutubeDownloader


class YoutubeVideoFaceDetector(VideoFaceDetector):
    @staticmethod
    def make_url_filename_safe(url):
        return re.sub('[^\w\-_\. ]', '_', url)

    def run(self, url, model_path='../output/model.xml'):
        filename = "videos/" + self.make_url_filename_safe(url) + ".mp4"
        YoutubeDownloader().download(url, filename)
        super().run(filename, model_path)


if __name__ == "__main__":
    url = input("Url: ")
    YoutubeVideoFaceDetector().run(url)
