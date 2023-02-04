import youtube_dl


# > This class is used to download videos from youtube
class YoutubeDownloader:
    def download(self, url, name):
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
    url = input("Url")
    name = input("Name")
    name = 'videos/' + name + '.mp4'
    YoutubeDownloader().download(url, name)
