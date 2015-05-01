import numpy as np

def rl_apply(v, func):
    import numpy
    if v is None or len(v)==0: return []
    #Get the sort index
    si  = numpy.argsort(v)

    #Use the sort index to create a sorted array
    sa  = v[si]

    #Find the bincounts using the unique boundaries of the array.
    mask = (sa[0:-1] != sa[1:])
    bounderies = numpy.nonzero(mask)[0]
    if bounderies.size==0:
        return numpy.repeat(func(v[0]), v.size)
    cnt = numpy.empty(bounderies.size+1, numpy.uint32)
    cnt[0] = bounderies[0] + 1
    cnt[1:-1] = bounderies[1:] - bounderies[0:-1]
    cnt[-1] = sa.size - bounderies[-1] -1

    #Compute the offsets for each bin then use that to calculate the vector of unique values.
    off = cnt.cumsum() - cnt
    u = numpy.array([func(d) if d else None for d in sa[off]])

    #Compute the inverse sort index which will tells us how to put a sorted array back
    #to it's original order.
    isi = numpy.empty(len(v))
    isi[si] = numpy.arange(len(v))

    #Expand the unique values by the count for each unique value then use the unique values to get it
    #back to it's original position. Afterwards, we convert back to a list which is expected by callers
    #like FramePandas.from_frame.
    return numpy.repeat(u, cnt)[isi.astype(int)]

def encode(*cols):
    cols = [np.asarray(x) for x in cols] + [x.mask for x in cols if hasattr(x, 'mask')]
    if cols[0].size==0: return np.zeros(0, np.int)
    mask = np.zeros(cols[0].size-1, np.bool)
    for col in cols:
        if col.dtype.kind=='f':
            mask |= np.absolute(col[0:-1] - col[1:]) > 0.000000000001
        else:
            mask |= (col[0:-1] != col[1:])
    bounderies = np.nonzero(mask)[0]
    if bounderies.size==0:
        return np.array([mask.size+1])
    cnt = np.empty(bounderies.size+1, np.uint32)
    cnt[0] = bounderies[0] + 1
    cnt[1:-1] = bounderies[1:] - bounderies[0:-1]
    cnt[-1] = cols[0].size - bounderies[-1] -1
    return cnt

def _expand_counts(*cnts):
    return cnts[0] if len(cnts)==1 and isinstance(cnts[0], list) else cnts

def decode(val, *cnts):
    cnts = _expand_counts(*cnts)
    prod = cnts[0].copy()
    for cntx in cnts[1:]:
        prod *= cntx
    return np.repeat(val, prod)

def fill(val, *cnts):
    cnts = _expand_counts(*cnts)
    return decode(val, *cnts)

def range(cnts, axis=0, roll=0, step=1):
    cnts = _expand_counts(*cnts)
    prod = cnts[0].copy()
    for cntx in cnts[1:]:
        prod *= cntx
    
    aprod = cnts[0].copy()
    for cntx in cnts[1:axis+1]:
        aprod *= cntx
    
    bprod = prod / aprod
    aprod_cumsum = np.cumsum(aprod)
    aprod_sum = 0 if aprod.size==0 else aprod_cumsum[-1]
    retval = np.repeat(np.arange(aprod_sum, dtype=np.uint32) - np.repeat((aprod_cumsum - aprod).astype(np.uint32), aprod),
                       np.repeat(bprod, aprod))
    
    if roll != 0:
        roll = roll * -1
        roll = np.zeros(cnts[axis].size, np.int32) + roll
        roll = fill(roll, *cnts)
        retval += roll
        
    return (retval % np.repeat(cnts[axis], prod)).astype(np.uint32)

def index(pos, *cnts, **kw):
    cnts = _expand_counts(*cnts)
    retval = decode(pos, *cnts) + range(cnts, **kw)
    return retval

def offsets(*cnts):
    cnts = _expand_counts(*cnts)
    return (end_offsets(*cnts) - cnts) + 1

def end_offsets(*cnts):
    cnts = _expand_counts(*cnts)
    return np.cumsum(cnts) - 1
