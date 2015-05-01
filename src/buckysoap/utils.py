import numpy as np

class one2many(dict):
    def reverse_lookup(self, x):
        if not hasattr(self, 'reverse'):
            reverse = dict()
            for k, v in self.items():
                for code in v.split():
                    reverse[code] = k
            self.reverse = reverse
        return np.unique([self.reverse[y] for y in x.split(',') if self.reverse.has_key(y)]).tolist()
    def __iter__(self):
        return (x for x in sorted(self.keys()))
