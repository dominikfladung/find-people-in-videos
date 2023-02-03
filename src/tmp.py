import time
import progressbar

bar = progressbar.ProgressBar(maxval=100)
bar.start()
for i in range(100):
    bar.update(i)
    time.sleep(0.02)
