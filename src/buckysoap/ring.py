import inspect
from series import Series

class Ring(object):
    def __init__(self, series, pos):
        self.ring = self
        self.series = series
        self.pos = pos
    def __getattr__(self, name):
        return getattr(self.series, name)
    def __getitem__(self, index):
        if isinstance(index, slice):
            start = 0 if index.start is None else index.start
            stop  = 0 if index.stop  is None else index.stop
            if start<0:
                start +=1
                stop  +=1
            return [self[i+start] for i in range(stop-start)]
        else:
            return self.series[self.pos+index]
    @property
    def date(self):
        return self.dates[self.pos]
    @property
    def datetime(self):
        return '%s 00:00:00' % (self.date)
    @property
    def year(self):
        return int(self.date[0:4])
    @property
    def month(self):
        return int(self.date[5:7])
    @property
    def day(self):
        return int(self.date[8:10])
    @property
    def YYYYMMDD(self):
        return int('%s%s%s' % (self.date[0:4], self.date[5:7], self.date[8:10]))
    @property
    def prev(self):
        return None if self.days_since_epoch<=0 else self[-1]
    @property
    def days_since_epoch(self):
        return self.pos - self.dates.searchsorted(self.epoch)

    series = classmethod(Series)

