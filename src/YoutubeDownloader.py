import youtube_dl


class YoutubeDownloader:
    def download(self, url, name):
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]',
            'outtmpl': 'videos/' + name + '.mp4',
        }
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])


if __name__ == "__main__":
    url = input("Url")
    name = input("Name")
    YoutubeDownloader().download(url, name)
