# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=3>

# Meet the Atom.

# <markdowncell>

# First let's create a few and prove that they are Atoms.

# <codecell>

import buckysoap as bs
print bs.zeros(10, int)
print bs.ones(10, int)
print bs.arange(10)
print type(bs.zeros(10, int))

# <markdowncell>

# The previous Atoms seem a lot like np.ndarray. This is because Atom just extends np.ndarray. Let's look at what functionality Atom adds to np.ndarray.

# <codecell>

x = bs.Atom.fromlist([[0, 1, 2, None, 4], None, [], [5, 6]])
print x
print x.asarray() #This np.ndarray portion of the data
print x.mask      #This exists on all Atoms to signify None
print x.bincounts #This represents the bincounts on each axis of our Atom

# <markdowncell>

# The Atom's depth is refered to as cardinality. This Atom has a cardinality of 3, for the previous it is 2 and for bs.zeros(10) it is 1.

# <codecell>

y = bs.Atom.fromlist([[[0, 1, 2, None, 4], None, [5, 6]], [[7, 8, 9, 4], [None, 10, 11]]])
print y
print y.asarray()
print y.mask
print y.bincounts
print bs.zeros(10).cardinality
print x.cardinality
print y.cardinality
print x.counts
print y.counts

# <markdowncell>

# Atoms support all binary operators.

# <codecell>

print x * 2
print x + 2
print y - 2
print y / 2

# <markdowncell>

# Atoms support methods like sum, average, stddev, tstat, ... The result of the functions always has a cardinality one less the that of it's source.

# <codecell>

print x
print x.sum()
print x.sum().sum()
print y
print y.sum()
print y.sum().sum()
print y.sum().sum().sum()
print y.average()

# <codecell>

print y.vstack(y)

# <markdowncell>

# Take note of what is happening here it is going to be instrumental to how Element.group works. Here z2.cardinality => 2 and y.cardinality => 3 therefore z2[y].cardinality => z2.cardinality + y.cardinality - 1 => 4. 

# <codecell>

z = bs.arange(12) + 100
print 'z =', z
print
print 'y =', y
print
print 'z[y] =', z[y]
print
z2 = bs.arange(78)
z2.bincounts.append((bs.arange(12) + 1))
print 'z2 =', z2
print
print 'z2[y] =', z2[y]
print z2.cardinality, y.cardinality, z2[y].cardinality

# <headingcell level=3>

# Combining Atoms to form an Element

# <markdowncell>


# <codecell>

e1 = bs.Element(name='Sample')(
    range = bs.arange(10),
    zero = bs.zeros(10, int),
    one = bs.ones(10, int))
e1.display()

# <markdowncell>


# <codecell>

e2 = e1(
    group_id = lambda x: (x.range % 3) * 2
)
e2.display()

# <markdowncell>


# <codecell>

e3 = e2.group('group_id')
e3.display()

# <markdowncell>


# <codecell>

e4 = e3(
    range_sum = lambda x: x.range.sum(),
    zero_sum = lambda x: x.zero.sum(),
    one_sum = lambda x: x.one.sum())
e4.display()
index = e3.__source__.__index__
print index
a = getattr(e3.__source__.__source__, 'range')
print a
print (a+100)[index]

# <markdowncell>


# <codecell>

e4 = e4('group_id,range,range_sum,zero,zero_sum,one,one_sum')
e4.display()
e3.display()
e2.display()
e1.display()

# <markdowncell>


# <codecell>

groups = (
    bs.Element(name='Groups')
    (group_id = bs.arange(6))
    (name = lambda x: ['group_%d' % y for y in x.group_id[::-1]]))
groups.display()
e4.display()

# <markdowncell>


# <codecell>

join = e4.inner(groups, group_id='group_id')
join.display()

# <markdowncell>


# <codecell>

join = join.sort_by('Groups.name').display()

# <codecell>

join.vstack(join).display()

# <markdowncell>


# <codecell>

print join.toxml()

# <markdowncell>


# <codecell>

print join.toxml(use_attributes=False)

# <markdowncell>


# <codecell>

print join.tojson()

# <codecell>

def factory():
    print "Running Factory"
    return bs.Element(name='Sample')(
        range = bs.arange(10),
        zero = bs.zeros(10, int),
        one = bs.ones(10, int))
bs.Element(
    name='sample',
    cnames='range,zero,one',
    cachedir='cache/sample',
    factory=factory
).display()

