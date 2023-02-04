"""
This class is used to download videos from YouTube
"""
import youtube_dl


class YoutubeDownloader:
    @staticmethod
    def download(url, name):
        """
        It downloads the video from the url and saves it as the name
        
        :param url: The URL of the video you want to download
        :param name: The name of the file to be downloaded
        """
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]',
            'outtmpl': name,
        }
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])


if __name__ == "__main__":
    input_url = input("Url")
    input_name = input("Name")
    input_name = 'videos/' + input_name + '.mp4'
    YoutubeDownloader().download(input_url, input_name)
