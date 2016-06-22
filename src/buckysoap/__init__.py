from atom import Atom
from element import *
from ring import Ring
from field import Field
from timer import Timer

def ones(*a, **kw):
    return Atom(np.ones(*a, **kw))

def zeros(*a, **kw):
    return Atom(np.zeros(*a, **kw))

def arange(*a, **kw):
    return Atom(np.arange(*a, **kw))
