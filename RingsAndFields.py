# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import util.buckysoap as bs

class A(bs.Ring):

    dates = [str(x) for x in bs.arange(10) + 20100101]

    @property
    def x(self):
        return 'x[%s]' % self.date
    
    class B(bs.Field):

        @property
        def y(self):
            return 'y[%s]' % self.date
            
        @property
        def z(self):
            return '(%s + %s)' % (self.x, self.y)
        
    b = property(B)
        
series = A.series(horizons=(0,-21))
print series['20100107'].b.z
for o in series:
    print o.date, o.x, o.b.y, o.b.z

# <codecell>

class A(A):
    
    class B(A.B):

        @property
        def y(self):
            return 'y[%s]`' % self.date
    
    b = property(B)

series = A.series(horizons=(-1, 43))
print series['20100108'].b.z
for o in series:
    print o.date, o.x, o.b.y, o.b.z

# <codecell>

class A(A):
    
    dates = [str(x) for x in bs.arange(100) + 20100101]

    class B(A.B):

        def change(self, func):
            ring = self.ring
            horizons = ring.horizons
            values = [func(ring[x]) for x in horizons]
            return ((values[1] / values[0]) - 1) * 100
    
    b = property(B)

series = A.series(horizons=(-1, 2))
print series['20100108'].b.z
for o in series:
    if o.date < '20100153':
        print o.date, o.b.change(lambda x: float(x.date))

