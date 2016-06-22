import time

class Timer:
    def __init__(self, title):
        self.title = title

    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start
        print "%s ran in %s seconds" % (self.title, self.interval)
