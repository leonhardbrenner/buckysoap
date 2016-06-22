import numpy as np, run_length as rl

from atom import Atom
from lazy import lazy
from element import getcolumns, sort_index

class join(object):

    def __init__(self, left, right, keys, left_outer=False, right_outer=False):
        self.left = join.Side(self, left, left_outer, 0, *keys.values())
        self.right = join.Side(self, right, right_outer, 1, *keys.keys())
        self.left.other, self.right.other = self.right, self.left
        joined_columns = [np.concatenate([self.left.unique[i], self.right.unique[i]])
                          for i in xrange(0, len(keys))]
        _sort_index = np.lexsort(joined_columns)
        _bin_counts = rl.encode(*[x[_sort_index] for x in joined_columns])
        self.side_indexes = Atom(np.concatenate([np.arange(self.left.bin_counts.size),
                                                   np.arange(self.right.bin_counts.size)])[_sort_index],
                                   bincounts=[_bin_counts]) 
    @lazy
    def inners(self):
        return self.side_indexes[self.side_indexes.bincounts[-1]==2]
    @lazy
    def mask(self):
        mask = np.ones(max(self.left.new_index.size, self.right.new_index.size), np.bool)
        for side_mask in (side.mask for side in (self.left, self.right) if side.mask is not None):
            mask &= side_mask.astype(bool)
        return mask
    
    class Side(object):
        def __init__(self, join, container, outer, axis, *keys):
            self.size           = len(container)
            self.join           = join
            self.axis           = axis
            self.outer          = outer
            _sort_index         = sort_index(container, *keys)
            self.columns        = getcolumns(container[_sort_index], *keys)
            self.bin_counts     = rl.encode(*self.columns)
            self.unique         = [Atom(column, bincounts=[self.bin_counts]).first
                                   for column in self.columns]
            self.inverted_index = Atom(_sort_index, bincounts=[self.bin_counts])
        @lazy
        def new_lengths(self):
            lengths = [self.outer_lengths, self.inner_lengths,
                       np.ones(self.other.outer_lengths.size, np.int)]
            return np.concatenate(lengths[::(1 if self.axis==0 else -1)])
        @lazy
        def new_index(self):
            #We add -1 to align with the other sides values
            def raw_index():
                raw_index = [
                    self.outer_index.flattened,
                    self.inner_index.flattened,
                    np.zeros(self.other.outer_lengths.size, np.int) - 1
                ]
                return np.concatenate(raw_index[::(1 if self.axis==0 else -1)])
            raw_index = raw_index()
            left, right = self.join.left, self.join.right
            rl_index = rl.index(
                np.cumsum(self.new_lengths) - self.new_lengths,
                left.new_lengths, right.new_lengths,
                axis=self.axis)
            #print rl_index[0:200]
            index = raw_index[rl_index]          
            return index
        @lazy
        def inner_ids(self):
            return self.join.inners.first if self.axis==0 else self.join.inners.last
        @lazy
        def inner_index(self):
            return self.inverted_index[self.inner_ids]
        @lazy
        def inner_lengths(self):
            return self.inner_index.bincounts[-1]
        @lazy
        def outer_index(self):
            mask = np.ones(self.inverted_index.size, np.int)
            mask[self.inner_ids] -= 1
            return self.inverted_index[np.nonzero(mask)[0]]
        @lazy
        def outer_lengths(self):
            return self.outer_index.bincounts[-1]
        @lazy
        def mask(self):
            return np.ones(self.new_index.size, np.int) if self.other.outer else self.new_index>=0
