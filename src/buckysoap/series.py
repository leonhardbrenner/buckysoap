class Series(object):
    def __init__(self, ring, **kw):
        self.ring = ring
        self(**kw)
    def __getitem__(self, index):
        if isinstance(index, str):
            return [self[i] for i, date in enumerate(self.ring.dates) if date==index][0]
        else:
            return self.ring(self, index)
    def __len__(self):
        return len(self.ring.dates)
    def __iter__(self):
        return (self.ring(self, i) for i in range(len(self)))
    def __call__(self, **kw):
        for key, value in kw.items():
            setattr(self, key, value)
        return self
