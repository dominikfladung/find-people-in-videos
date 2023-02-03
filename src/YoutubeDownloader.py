import youtube_dl


class YoutubeDownloader:
    def download(self, url, name):
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
