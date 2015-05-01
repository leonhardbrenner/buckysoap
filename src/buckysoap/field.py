class Field(object):
    def __init__(self, ring):
        self.ring = ring
    def __getattr__(self, name):
        return getattr(self.ring, name, None)
