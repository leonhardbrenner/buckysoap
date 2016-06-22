class Field(object):
    def __init__(self, ring):
        self.ring = ring
    def __getattr__(self, name):
        return getattr(self.ring, name, None)
    def __call__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
        return self
