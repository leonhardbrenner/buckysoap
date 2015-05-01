import datetime
import os
import traceback
import numpy as np
import run_length as rl

from lockfile import FileLock
from lru_cache import lru_cache

class Atom(np.ndarray):
    class __cache__(object):
        @lru_cache(maxsize=100)
        def fetch(self, filename, is_client):
            #print 'Loading %s' % filename
            args = dict()
            lock = FileLock(filename)
            bincounts = []
            def load():
                npzfile = np.load(lock.path)
                for f in npzfile.files:
                    if f.startswith('bincount'):
                        bincounts.append((lambda x: Atom(x, mask=x!=-1))(npzfile[f]))
                    else:
                        args[f] = npzfile[f]
                npzfile.close()
                args['bincounts'] = bincounts
            if is_client:
                load()
            else:
                with lock:
                    load()
            return Atom(**args)
    __cache__ = __cache__()

    EMPTY = 0
    @classmethod
    def is_cached(cls, cachedir, name):
        return os.path.exists('%s/%s.npz' % (cachedir, name))
    @classmethod
    def load(cls, cachedir, name, is_client=False):
        #traceback.print_stack()
        filename = '%s/%s.npz' % (cachedir, name)
        return Atom.__cache__.fetch(filename, is_client)
    def persist(self, cachedir, name):
        #print 'Persisting %s/%s.npz' % (cachedir, name)
        args = dict()
        args['data'] = np.asarray(self)
        args['mask'] = self.mask
        for i, bincount in enumerate(self.bincounts):
            args['bincount%s' % i] = bincount.asarray()
            args['bincount%s' % i][~bincount.mask] = -1
        flock = FileLock('%s/%s.npz' % (cachedir, name))
        with flock:
            if not os.path.exists(flock.path):
                np.savez_compressed(flock.path, **args)
    def __new__(cls,
                data          = None,
                mask          = None,
                bincounts     = [],
                unique        = None,
                unique_counts = None,
                sort_index    = None,
                type          = None,
                cachedir      = None,
                name          = None,
                ):
        type2dtype = {
            bool              : np.bool,
            float             : np.float,
            int               : np.int,
            long              : np.long,
            str               : np.str,
            datetime.date     : np.datetime64,
            datetime.datetime : np.datetime64,
            }
        #np.array([x if x is not None else datetime.datetime(1,1,1) for x in y], dtype=np.datetime64)
        if data is None:
            self = np.repeat((unique), (unique_counts))
            if sort_index is not None:
                inverse_sort_index = np.empty(sort_index.size, np.int)
                inverse_sort_index[sort_index] = np.arange(sort_index.size)
                self = np.asarray(self)[inverse_sort_index]
        else:
            if isinstance(data,list) and None in data:
                if mask is None: mask = [ x is not None for x in data ]
                data = [ cls.EMPTY if x is None else x for x in data ]
            self = np.asarray(data, dtype=None if type is None else type2dtype[type])
            
        if mask is not None:
            mask = np.asarray(mask, np.bool)
            self[~mask] = cls.EMPTY
        else:
            mask = np.ones(self.size, np.bool)
            
        self = self.view(cls)

        if mask is not None:
            self.mask = mask

        if bincounts is not None:
            self.bincounts = [bincount if isinstance(bincount, Atom) else Atom(bincount) for bincount in bincounts]
        self.type = type
        return self
    def __array_finalize__(self, obj):
        mask = np.ones(obj.size, np.bool) if obj is not None and not hasattr(obj, "mask") else None
        self.mask = getattr(obj, 'mask', mask)
        self.bincounts = getattr(obj, 'bincounts', [])
    def __str__(self):
        return str(self.tolist())
    @classmethod
    def fromlist(cls, src):
        def walk(src, context, depth):
            if len(context) == depth:
                context.append(dict(data=[], mask=[]))
            if src is None:
                context[depth]['mask'].append(False)
                context[depth]['data'].append(0)
            else:
                context[depth]['mask'].append(True)
                if isinstance(src, list):
                    context[depth]['data'].append(len(src))
                    if len(src)==0 and len(context)==depth+1:
                        context.append(dict(data=[], mask=[]))
                    for x in src:
                        walk(x, context, depth+1)
                else:
                    context[depth]['data'].append(src)
            return context
        context = walk(src, [], 0)
        return cls(context[-1]['data'],
                   mask=context[-1]['mask'],
                   bincounts=[Atom(**x) for x in context[1:-1][::-1]])
    def tolist(self):
        retval = [b if m else None for b, m in zip(np.asarray(self),
                                                   self.mask if self.mask is not None else np.ones(self.size)) ]
        for bincounts in self.bincounts:
            offsets = bincounts.cumsum() - bincounts
            end_offsets = offsets + bincounts
            retval = [retval[start:end] if mask else None
                      for start, end, mask in zip(offsets, end_offsets, bincounts.mask)]
        return retval
    def __len__(self):
        if len(self.bincounts)==0:
            return np.asarray(self).__len__()
        else:
            return self.bincounts[-1].size
    @property
    def counts(self):
        return Atom(self.bincounts[0].asarray(), self.bincounts[0].mask, self.bincounts[1:])
    @property
    def size(self):
        return len(self)
    def asarray(self, masknum=0):
        array = np.asarray(self)
        if self.mask is not None:
            array[~self.mask] = 0
        return array
    def __getslice__(self, start=None, stop=None, step=None):
        return self[Atom(np.arange(len(self))[start:stop:step])]
    def __getitem__(self, index):
        oindex = index
        if isinstance(index, slice):
            return self.__getslice__(index.start, index.stop, index.step)
        if isinstance(index, list):
            return self[np.array(index, np.int)]
        if isinstance(index, int):
            if len(self.bincounts)==0:
                return None if not self.mask[index] else np.asarray(self)[index]
            elif not self.bincounts[-1].mask[index]:
                return None
        if len(self.bincounts)==0:
            return self.__getitem3__(oindex, index)
        else:
            return self.__getitem2__(oindex, index)
    def __getitem3__(self, oindex, index):
        return self.__class__(np.asarray(self)[index],
                              mask=self.mask[index]
                              & (index.mask if hasattr(index, 'mask') and index.dtype!=np.bool else True),
                              bincounts=index.bincounts[:] if hasattr(index, 'bincounts') else [])
    def __getitem2__(self, oindex, index):
        #import pdb; pdb.set_trace()
        bincounts = self.bincounts[:]
        if np.isscalar(index):
            b = bincounts[-1]
            if b.mask[index]==False: return None
            index = np.arange(b[index]) + (b.cumsum()-b)[index]
            bincounts = bincounts[:-1]
        nbc = []
        for x in bincounts[::-1]:
            b = x.asarray()
            m = x.mask[index] &(index.mask if hasattr(index, 'mask') and index.dtype != bool else True)
            nbc.append(Atom(b[index], mask=m))
            index = rl.index((b.cumsum() - b)[index][m], b[index][m])
        return Atom(
            self.asarray()[index],
            mask=self.mask[index],
            bincounts=nbc[::-1] + (oindex.bincounts if hasattr(oindex, 'bincounts') else [])) 
    @property
    def cardinality(self):
        return len(self.bincounts) + 1
    def leaves(self, start=None, stop=None, step=None):
        if len(self.bincounts)==0:
            return self[start:stop:step]
        else:
            new_offsets = np.asarray(((self.bincounts[0].cumsum() - 1).asarray() - self.bincounts[0]) + 1)
            new_bincounts = np.asarray(self.bincounts[0]).copy()
            if start is not None:
                new_offsets += start
                new_bincounts -= start
            if stop is not None:
                new_bincounts += stop
            index = rl.index(new_offsets, new_bincounts)
            index = index.astype(int)
            if step is not None:
                index = index[(rl.range(new_bincounts) % step)==0]
                new_bincounts /= step
            return self.__class__(np.asarray(self)[index],
                                  mask=self.mask[index],
                                  bincounts=[new_bincounts])
    def leaf(self, start):
        self = self[start<self.bincounts[0]]
        offsets = (self.bincounts[0].cumsum() - self.bincounts[0]) + start
        return self.__class__(np.asarray(self)[offsets],
                              mask=(self.mask[offsets] & self.bincounts[0].mask),
                              bincounts=self.bincounts[1:])
    def __iter__(self):
        for i in np.arange(self.size):
            yield self[i]
    def isnan(self):
        return np.isnan(np.asarray(self))
    def nan_to_mask(self):
        if not np.issubdtype(self.dtype, float):
            return self
        else:
            return self.__class__(np.nan_to_num(np.asarray(self)),
                                  mask=self.mask & ~np.isnan(np.asarray(self)),
                                  bincounts=self.bincounts)
    def nan_to_num(self):
        if not isinstance(self.dtype, np.float):
            return self
        else:
            return self.__class__(np.nan_to_num(np.asarray(self)),
                                  mask=self.mask,
                                  bincounts=self.bincounts)
    def purge(self, nans=True):
        condition = self.mask
        if nans: condition &= ~np.isnan(np.asarray(self))
        if len(self.bincounts)==0:
            return self[condition]
        else:
            bincounts = self.bincounts[:]
            bincounts[0] = Atom(condition, bincounts=[bincounts[0]]).cumsum().last
            return Atom(np.asarray(self)[condition], mask=self.mask[condition], bincounts=bincounts)
    def cumsum(self):
        self = self.nan_to_mask()
        cs = self.asarray().cumsum()
        if len(self.bincounts)==0:
            return self.__class__(cs, mask=self.mask)
        else:
            bincounts = self.bincounts[0].asarray()
            end_offsets = (bincounts.cumsum() - 1)[:-1].astype(int)
            cs[bincounts[0]:] -= np.repeat((cs[end_offsets] * (end_offsets>=0)), bincounts[1:])
            return self.__class__(cs, mask=self.mask, bincounts=self.bincounts)
    @property
    def first(self):
        if len(self.bincounts)==0:
            return self[0] if self.mask[0] else None
        else:
            offsets = (self.bincounts[0].cumsum() - self.bincounts[0]).astype(int)
            return self.__class__(np.asarray(self)[offsets],
                                  mask=(self.mask[offsets] & self.bincounts[0].mask),
                                  bincounts=self.bincounts[1:])
    @property
    def last(self):
        if len(self.bincounts)==0:
            return self[-1] if self.mask[-1] else None
        else:
            offsets = (self.bincounts[0].cumsum() - 1).astype(int)
            return self.__class__(self.asarray()[offsets],
                                  mask=(self.mask[offsets] & self.bincounts[0].mask),
                                  bincounts=self.bincounts[1:])
    def sum(self):
        sum = self.cumsum().last
        if len(self.bincounts)>0:
            sum[self.counts==0] = 0
        return sum
    def diff(self):
        return self.first - np.choose((self.bincounts[0]==1).asarray(), [self.last, self.zeros])

    @property
    def sort_index(self):
        array = self.asarray()
        if array.dtype == np.datetime64:
            sort_index = np.argsort(array)
            inverse_sort_index = np.empty(len(array), dtype=int)
            inverse_sort_index[sort_index] = np.arange(len(array))
            sorted_array = array[sort_index]
            bincounts = rl.encode(sorted_array)
            array = np.repeat(np.array([int(re.sub('[-: ]', '', str(x)))
                                        for x in sorted_array[bincounts.cumsum() - bincounts]]),
                              bincounts)[inverse_sort_index]
        if len(self.bincounts)==0:
            sort_index = np.lexsort([self.mask, array])
            return Atom(sort_index,
                        mask = self.mask[sort_index])
        else:
            bincounts = np.asarray(self.bincounts[0]).astype(np.int)
            bincounts[~self.bincounts[0].mask] = 0
            binid = np.repeat(np.arange(bincounts.size), bincounts)
            sort_index = np.lexsort([self.mask, array, binid])
            return Atom(sort_index,
                        mask=self.mask[sort_index],
                        bincounts=self.bincounts)
    @property
    def sort_index(self):
        array = self.asarray()
        if array.dtype == np.datetime64 or array.dtype == object:
            #import pdb; pdb.set_trace()
            #sort_index = np.argsort(array)
            #inverse_sort_index = np.empty(len(array), dtype=int)
            #inverse_sort_index[sort_index] = np.arange(len(array))
            #sorted_array = array[sort_index]
            #bincounts = rl.encode(sorted_array)
            #array = np.repeat(np.array([int(re.sub('[-: ]', '', str(x)))
            #                            for x in sorted_array[bincounts.cumsum() - bincounts]]),
            #                  bincounts)[inverse_sort_index]
            def convert(x):
                import re
                import decimal
                if isinstance(x, np.datetime64):
                    return int(re.sub('[-: ]', '', str(x)))
                elif isinstance(x, decimal.Decimal):
                    return str(x)
                else:
                    return str(x)
            array = np.array([convert(x) for x in array])
        if len(self.bincounts)==0:
            return Atom(np.lexsort([self.mask, array]))
        else:
            bincounts = np.asarray(self.bincounts[0]).astype(np.int)
            bincounts[~self.bincounts[0].mask] = 0
            binid = np.repeat(np.arange(bincounts.size), bincounts)
            sort_index = np.lexsort([self.mask, array, binid])
            return Atom(sort_index, mask=self.mask[sort_index], bincounts=self.bincounts)
    @property
    def inverse_sort_index(self):
        index = np.empty(len(self), np.int)
        index[self.sort_index] = np.arange(len(self))
        return Atom(index, mask=self.mask[index], bincounts=self.bincounts)
    rank = inverse_sort_index
    @property
    def sorted(self):
        sort_index = self.sort_index
        return self.__class__(np.asarray(self)[np.asarray(sort_index)],
                              mask=self.mask[np.asarray(sort_index)],
                              bincounts=sort_index.bincounts)
    @property
    def unique_index(self):
        from element import Element
        if len(self.bincounts)==0:
            sorted = self.sorted
            bincounts = rl.encode(sorted.mask, self.sorted)
            offsets = np.cumsum(bincounts) - bincounts
            return self.sort_index[offsets]
        else:
            bincounts = np.asarray(self.bincounts[0]).astype(np.int)
            bincounts[~self.bincounts[0].mask] = 0
            binid = np.repeat(np.arange(bincounts.size), bincounts)[self.sort_index.flattened]
            sorted = self.sorted.flattened

            array = sorted[:]
            array[~sorted.mask] = 0
            bincounts = rl.encode(binid, sorted.mask, array)
            offsets = np.cumsum(bincounts) - bincounts

            bincounts2 = rl.encode(binid[offsets])
            offsets2 = np.cumsum(bincounts2) - 1

            bincounts3 = np.zeros(self.bincounts[0].size, np.int)
            bincounts3[binid[offsets][offsets2]] = bincounts2

            bincounts3 = [Atom(bincounts3, mask=self.bincounts[0].mask)]
            bincounts3.extend(self.bincounts[1:])
            return Atom(self.sort_index.flattened[offsets],
                        mask=sorted.mask[offsets],
                        bincounts=bincounts3)
    @property
    def unique(self):
        unique_index = self.unique_index
        return self.__class__(np.asarray(self)[np.asarray(unique_index)],
                              mask=self.mask[np.asarray(unique_index)],
                              bincounts=unique_index.bincounts)
    @property
    def unique_counts(self):
        return rl.encode(self.sorted)
    def flatten(self):
        bincounts = []
        if len(self.bincounts) > 1:
            bincounts.append(self.__class__(self.bincounts[0],
                                            mask=self.bincounts[0].mask,
                                            bincounts=[self.bincounts[1]]).sum())
            bincounts.extend(self.bincounts[2:])
        return self.__class__(np.asarray(self), mask=self.mask, bincounts=bincounts)
    @property
    def flattened(self):
        return Atom(np.asarray(self), mask=self.mask)
    @property
    def min(self):
        return self.sorted.first
    @property
    def max(self):
        return self.sorted.last
    @property
    def inverted_index(self):
        return Atom(self.sort_index, bincounts=[self.unique_counts])
    def add_mask(self, mask):
        return self.__class__(self, mask=mask, bincounts=self.bincounts)
    def vstack(self, *columns):
        new_self = [np.asarray(self)]
        new_mask = [self.mask]
        new_bincounts = [[bincounts[:]] for bincounts in self.bincounts]
        for column in columns:
            new_self.append(np.asarray(column))
            new_mask.append(column.mask)
            for i, bincounts in enumerate(column.bincounts):
                new_bincounts[i].append(bincounts[:])
        return self.__class__(np.hstack(new_self),
                              mask      = np.hstack(new_mask),
                              bincounts = [bincounts[0].vstack(*bincounts[1:])
                                           for bincounts in new_bincounts])
    def hstack(self, *columns):
        ncols = len(columns)+1
        v = self.vstack(*columns)[np.arange(self.size * ncols).reshape(ncols, self.size).transpose().flatten()]
        v.bincounts.append(Atom(np.zeros(self.size, np.int) + ncols))
        return v
    def average(self):
        x = self.purge()
        if len(self.bincounts)==0:
            return x.sum() / len(x)
        elif len(self.bincounts)==1:
            return self.sum() / x.bincounts[0]
        else:
            return self.sum() / Atom(x.bincounts[0].asarray(), x.bincounts[0].mask, x.bincounts[1:])
    def stddev(self):
        if len(self.bincounts)==0:
            return np.std(self[self.mask].asarray())
        else:
            self = self.purge()
            return ((self**2).sum()/self.bincounts[0] - self.average()**2).sqrt()
    def __tstat__(self):
        if len(self.bincounts)==0:
            return self.average() / (self.stddev() / np.sqrt(self[self.mask].size))
        else:
            return self.average() / (self.stddev() / self.purge().bincounts[0].sqrt())
    tstat = __tstat__
    def __sqrt__(self):
        return self.__class__(np.sqrt(self.asarray()), mask=self.mask, bincounts=self.bincounts)
    sqrt = __sqrt__
    def __log__(self):
        return self.__class__(np.log(self.asarray()), mask=self.mask, bincounts=self.bincounts)
    log = __log__
    @property
    def absolute(self):
        return self.__class__(np.absolute(self.asarray()), mask=self.mask, bincounts=self.bincounts)
    def demean(self):
        return self - self.average()
    def wrap_ufunc(self, name, other):
        def align(x, y):
            if np.isscalar(x):
                x = Atom(np.repeat(x, len(y)))
            if np.isscalar(y):
                y = Atom(np.repeat(y, len(x)))
            if not isinstance(x, Atom):
                x = Atom(x)
            if not isinstance(y, Atom):
                y = Atom(y)
            counts = [len(bc) for bc in x.bincounts[::-1]]
            for i, bc in enumerate(y.bincounts[::-1]):
                if len(bc) in counts: continue
                x = Atom(np.repeat(x.asarray(), bc.asarray()),
                           mask=np.repeat(x.mask, bc.asarray()),
                           bincounts=y.bincounts[0:i+1])
            return x
        self, other = align(self, other), align(other, self)
        return self.__class__(getattr(self.asarray(), name)(other.asarray()),
                              mask=(True if self.mask is None else self.mask) & other.mask,
                              bincounts=self.bincounts)
    def __eq__(self, other):
        return self.wrap_ufunc('__eq__', other)
    def __ne__(self, other):
        return self.wrap_ufunc('__ne__', other)
    def __lt__(self, other):
        return self.wrap_ufunc('__lt__', other)
    def __le__(self, other):
        return self.wrap_ufunc('__le__', other)
    def __ge__(self, other):
        return self.wrap_ufunc('__ge__', other)
    def __gt__(self, other):
        return self.wrap_ufunc('__gt__', other)
    def __add__(self, other):                         
        return self.wrap_ufunc('__add__', other)
    def __sub__(self, other):                         
        return self.wrap_ufunc('__sub__', other)
    def __mod__(self, other):                         
        return self.wrap_ufunc('__mod__', other)
    def __mul__(self, other):                         
        return self.wrap_ufunc('__mul__', other)
    def __div__(self, other):                         
        return (self*1.0).wrap_ufunc('__div__', other)
    def __truediv__(self, other):                         
        return (self*1.0).wrap_ufunc('__truediv__', other)
    def __iadd__(self, other):                         
        return self.wrap_ufunc('__add__', other)
    def __isub__(self, other):                         
        return self.wrap_ufunc('__sub__', other)
    def __imod__(self, other):                         
        return self.wrap_ufunc('__mod__', other)
    def __imul__(self, other):                         
        return self.wrap_ufunc('__mul__', other)
    def __idiv__(self, other):                         
        return self.wrap_ufunc('__div__', other)
    def split(self, delim):
        return Atom.fromlist([x.split(delim) for x in self])
    def lag(self, n, sep=None):
        if len(self.bincounts)==0:
            return self
        else:
            bincounts = self.bincounts[-1].asarray()
            offsets = ((bincounts.cumsum() - 1) - bincounts) + 1
            lag_bincounts = bincounts-n+1
            lag_bincounts[lag_bincounts<0] = 0
            new_bincounts = np.repeat(np.repeat(n, len(bincounts)), lag_bincounts)
            array = self.asarray()[rl.index(rl.index(offsets, lag_bincounts), new_bincounts)]
            if sep is None:
                return Atom(array, bincounts=self.bincounts[0:-1] + [new_bincounts] + [lag_bincounts])
            else:
                concat = np.core.defchararray.add
                r = array[0::n]
                for i in range(1, n):
                    r = concat(concat(r, ' '), array[i::n])
                return Atom(r, bincounts=self.bincounts[0:-1] + [lag_bincounts])
    def join(self, sep=' '):
        if len(self.bincounts)==0:
            return sep.join([str(y) for y in self])
        else:
            return Atom.fromlist([None if x is None else x.join(sep) for x in self])
    @property
    def arange(self):
        if len(self.bincounts)==0:
            return Atom(np.arange(len(self)))
        else:
            return Atom(data=rl.range([self.bincounts[-1].asarray()]), mask=self.mask, bincounts=self.bincounts)
    @property
    def zeros(self):
        return Atom(np.zeros(len(self)))
    @property
    def ones(self):
        return Atom(np.ones(len(self), np.int))
    def startswith(self, y):
        return Atom([x.startswith(y) for x in self])
    def endswith(self, y):
        return Atom([x.endswith(y) for x in self])
    def match_all(self, *a):
        matches = self.zeros
        for x in a:
            matches+=self==x
        return matches.sum()==len(a)
    def match_any(self, *a):
        matches = self.zeros
        for x in a:
            matches+=self==x
        return matches.sum()>1
