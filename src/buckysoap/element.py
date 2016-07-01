import os
import sys
import numpy as np
from atom import Atom
from datetime  import datetime

def by(self, *a):
    if len(a)==0:
        return self.__cnames__
    else:
        return a[0].split(',') if len(a)==1 else a

def group_index(self, *a):
    '''
    '''
    from atom import Atom
    import run_length as rl
    cnames = by(self, *a)
    sorted_self = sort_by(self, *cnames)
    sorted_columns = getcolumns(sorted_self, *cnames)
    return Atom(sort_index(self, *cnames),
                bincounts=[rl.encode(*sorted_columns)])

def sort_index(self, *a):
    cnames = by(self, *a)
    sort_index = None
    for name, col in zip(cnames, getcolumns(self, *cnames[::-1])):
        sort_index = col.sort_index if sort_index is None else sort_index[col[sort_index].sort_index]
    return sort_index

def sort_by(self, *a):
    cnames = by(self, *a)
    element = self[sort_index(self, *cnames)]
    element.__by__ = cnames
    return element

def getcolumn(self, path):
    attr = self
    for cname in path.split('.'):
        attr = getattr(attr, cname)
    return attr

def getcolumns(self, *paths):
    if len(paths)==1: paths = paths[0].split(',')
    return [getcolumn(self, path) for path in paths]

def namedtuples_to_element(*namedtuples):
    return Element()(
        **dict(zip(namedtuples[0]._fields,
                   [[getattr(row, name) for row in namedtuples]
                    for name in namedtuples[0]._fields])))

def join(self, right, left_outer=False, right_outer=False, **keys):
    '''
    '''
    from atom import Atom
    from join import join
    new = self
    for right in right if isinstance(right, list) else [right]:
        name = right.__name__
        left_index = None
        right_index = None
        mask = None
        index = join(self, right, keys, left_outer, right_outer)
        right = Element(source=right,
                        cnames=[x for x in right.__cnames__
                                if x not in keys.values()],
                        index=Atom(index.right.new_index[index.mask],
                                   mask=(index.right.new_index>=0)[index.mask]))
        new = Element(source = new,
                      index  = Atom(index.left.new_index[index.mask],
                                    mask=(index.left.new_index>=0)[index.mask]),
                      cnames = new.__cnames__ + [name])
        setattr(new, name, right)
    return new

def inner(self, right, **keys):
    '''
    '''
    return join(self, right, False, False, **keys)

def left_outer(self, right, **keys):
    '''
    '''
    return join(self, right, True, False, **keys)

def right_outer(self, right, **keys):
    '''
    '''
    return join(self, right, False, True, **keys)

def outer(self, right, **keys):
    '''
    '''
    return join(self, right, True, True, **keys)

def expand(self, *a):
    '''
    '''
    cnames = by(self, *a)
    bincounts = np.asarray(getattr(self, cnames[0]).bincounts[0])
    new = Element(source=self, index=np.repeat(np.arange(len(bincounts)), bincounts))
    #import pdb; pdb.set_trace()
    for cname in self.__cnames__:
        if cname in cnames:
            setattr(new, cname, getattr(self, cname).flatten())
    return new

def demean(self, val, *paths):
    '''
    '''
    group_index = self.__group_index__(*paths)
    inverse_sort_index = np.empty(group_index.flattened.size, np.int)
    inverse_sort_index[group_index.flattened] = np.arange(group_index.flattened.size)
    return val[group_index].demean().flattened[inverse_sort_index]

def distribution(self, val, *paths):
    '''
    '''
    group_index = self.__group_index__(*paths)
    inverse_sort_index = np.empty(group_index.flattened.size, np.int)
    inverse_sort_index[group_index.flattened] = np.arange(group_index.flattened.size)
    return val[group_index].distribution().flattened[inverse_sort_index]

def vstack(self, *elements):
    '''
    '''
    elements = [element
                for element in elements
                if element is not None]
    class View(object):
        __cnames__ = self.__cnames__
        def __getattr__(inner, name):
            attr = getattr(self, name)
            if len(elements)==0:
                return attr
            else:
                return attr.vstack(*[getattr(element, name)
                                     for element in elements])
    return Element(factory=View, name=self.__name__, cnames=self.__cnames__)

def freeze(self, *a):
    '''
    '''
    cnames = by(self, *a)
    return Element(cnames  = cnames,
                   columns = getcolumns(self, *cnames))

def keys(self, *keys):
    '''
    '''
    return Element(self, keys=[k for k in keys])

def alias(self, name):
    return Element(self, name=name)

def unique(self, *by):
    return first(group(self, *by))

def first(self):
    return Element(source=self.__source__, index=self.__index__.first) if self.__index__ is not None else self[0]

def last(self):
    return Element(source=self.__source__, index=self.__index__.last) if self.__index__ is not None else self[-1]

def arange(self):
    '''
    '''
    return Atom(np.arange(len(self)))

def zeros(self):
    '''
    '''
    return Atom(np.zeros(len(self)))

def ones(self):
    '''
    '''
    return Atom(np.ones(len(self)))

def toxml(obj, interface, pretty=False, indent='\t'):
    return getattr(interface, '__toxml__')(obj, pretty, indent)

def fromxml(xml, interface):
    return getattr(interface, '__fromxml__')(xml)

def element_to_dataframe(self):
    '''
    '''
    from pandas import DataFrame
    r = DataFrame({ c: getattr(self, c) for c in self.__cnames__ }, columns=self.__cnames__)
    return r
to_pandas = element_to_dataframe

def to_csv(self, f, index=False, header=True):
    self.to_pandas().to_csv(f, index=index, header=header)

def set_name(self, name):
    return Element(self, name=name)

def attrs(self, **new_columns):
    cnames = self.__cnames__[:]
    new    = Element(source=self)
    for name, val in new_columns.items():
        if isinstance(val, str):
            attr = getcolumn(new, val)
        elif hasattr(val, '__call__'):
            val = val(self)
            if isinstance(val, list):
                val = Atom.fromlist(val)
            attr = val
        elif isinstance(val, list):
            attr = Atom.fromlist(val)
        else:
            attr = val
        setattr(new, name, attr)
        if name not in cnames:
            cnames.append(name)
    new.__cnames__ = cnames
    return new

def to_string(self):
    from cStringIO import StringIO
    os = StringIO()
    self.display(output_stream=os)
    string = os.getvalue()
    os.close()
    return string

def group(self, *a):
    '''
    '''
    x = self[group_index(self, *a)]
    for cname in by(self, *a):
        setattr(x, cname, getcolumn(x, cname).first)
        #x = x(**{cname.split('.')[-1]:getcolumn(x, cname).first})
    return x

def display(self, n=20, indent = '    ', output_stream = sys.stdout, file=None, filename=None):
    '''
    '''
    if file is not None:
        filename = file
    if filename is not None:
        file = filename
    
    __indent = indent
    class Path(object):
        def __init__(self, parent, element, name, pos=None):
            self.parent = parent
            self.element = element
            self.name = name
            self.pos = None
        def __str__(self):
            return '.'.join(reversed(list(parent_iterator(self, lambda x: x.name))))

    def display(tag, node, parent=None, indent = ''):
        if isinstance(node, Element):
            if isinstance(getattr(node, node.__cnames__[0]), np.ndarray):
                for i, x in enumerate(node):
                    path = Path(parent, node, tag, i)
                    output_stream.write(indent + '%s[%d]' % (tag, i) + "\n")
                    for cname in x.__cnames__:
                        display(cname, getattr(x, cname), path, indent + __indent)
            else:
                path = Path(parent, node, tag)
                output_stream.write(indent + '%s' % (tag) + "\n")
                for cname in node.__cnames__:
                    display(cname, getattr(node, cname), path, indent + __indent)
        else:
            node_type = type(node)
            path = Path(parent, node, tag)
            output_stream.write(indent + '%s=%s' % (tag, node) + "\n")

    is_frame_like = True
    for name in self.__cnames__:
        attr = getattr(self, name)
        if not isinstance(attr, Atom) or len(attr.bincounts)>0:
            is_frame_like = False
    if is_frame_like:
        columns = zip(*[self.__cnames__]
                      + zip(*[(getattr(self, name)[0:n])
                              for name in self.__cnames__]))
        column_widths = [max([len(str(y)) for y in x]) for x in columns]
        column_format = [''.join(['%', str(x), 's']) for x in column_widths]
        row_format = ' '.join(column_format)
        for row in zip(*columns):
            print row_format % row
        #import pdb; pdb.set_trace()
        #print self[0:n].to_pandas().to_string()
    else:
        if filename is not None:
            output_stream = open(filename, 'w')
        display('x' if hasattr(self, '__name__') and self.__name__ is None else self.__name__, self[0:n])
    print "(%s rows)" % len(self)
    return self

def toxml(self, indent='    ', pretty=True, use_attributes=True, root=None):
    '''
    '''
    import lxml.etree as etree
    from atom import Atom
    from xml.dom import minidom
    import numpy as np
    def prettify(elem):
        """Return a pretty-printed XML string for the Element.
        """
        rough_string = etree.tostring(elem)
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent=indent)
    #Make root from arguments
    stack = [root if root is not None else etree.Element('root')]
    def display(tag, node, stack):
        if isinstance(node, Element):
            if isinstance(getattr(node, node.__cnames__[0]), np.ndarray):
                for i, x in enumerate(node):
                    stack.append(etree.SubElement(stack[-1], node.__name__))
                    for cname in x.__cnames__:
                        display(cname, getattr(x, cname), stack)
                    stack.pop()
            else:
                stack.append(etree.SubElement(stack[-1], node.__name__))
                for cname in node.__cnames__:
                    display(cname, getattr(node, cname), stack)
                stack.pop()
        else:
            value = node.join(' ') if isinstance(node, Atom) else str(node)
            if use_attributes:
                stack[-1].set(tag, value)
            else:
                etree.SubElement(stack[-1], tag).text = value

    display('x' if hasattr(self, '__name__') and self.__name__ is None else self.__name__, self, stack)
    def parent_iterator(link, func=lambda x: x, parent=lambda x: x.parent):
        while link is not None:
            yield func(link)
            link = parent(link)
    xml = etree.tostring([x for x in parent_iterator(stack[0], parent=lambda x: x.getparent())][-1])
    if pretty:
        return minidom.parseString(xml).toprettyxml(indent=indent)
    else:
        return xml

def tojson(self, indent='    ', pretty=True):
    '''
    '''
    from collections import OrderedDict
    import json
    from atom import Atom

    stack = [OrderedDict()]
    def display(tag, node, stack):
        if isinstance(node, Element):
            if isinstance(getattr(node, node.__cnames__[0]), Atom):
                for i, x in enumerate(node):
                    if node.__name__ not in stack[-1]:
                        stack[-1][node.__name__] = []
                    new = OrderedDict()
                    stack[-1][node.__name__].append(new)
                    stack.append(new)
                    for cname in x.__cnames__:
                        display(cname, getattr(x, cname), stack)
                    stack.pop()
            else:
                new = OrderedDict()
                stack[-1][node.__name__] = new
                stack.append(new)
                for cname in node.__cnames__:
                    display(cname, getattr(node, cname), stack)
                stack.pop()
        else:
            stack[-1][tag] = node.tolist() if isinstance(node, Atom) else str(node)
    display('x' if hasattr(self, '__name__') and self.__name__ is None else self.__name__, self, stack)
    if pretty:
        return json.dumps(stack[0], sort_keys=False, indent=4, separators=(',', ': '))
    else:
        return json.dumps(stack[0], sort_keys=False, separators=(',', ': '))

def repeat(self, x):
    return Atom(np.repeat(x, len(self)))

class Element(object):
    def __init__(self, source=None, factory=None, cachedir=None, index=None,
                 cnames=None, name=None, columns=None, is_client=True, kw={}):
        if cachedir and not os.path.exists(cachedir):
            os.makedirs(cachedir)
        self.__cachedir__ = cachedir
        self.__source__ = source
        self.__factory__ = factory
        self.__index__ = index

        #cnames and source.cnames
        def __cnames():
            __cnames = None
            if cnames:
                if hasattr(cnames, '__call__'):
                    __cnames = cnames(self)
                elif isinstance(cnames, str):
                    __cnames = cnames.split(',')
                else:
                    __cnames = cnames
            else:
                __cnames = []
                if source is not None:
                    __cnames += source.__cnames__
                if kw:
                    __cnames += [x for x in kw.keys() if x not in __cnames]
            return __cnames
        self.__cnames__ = __cnames()

        if columns is not None:
            for name, value in zip(self.__cnames__, columns):
                setattr(self, name, value)

        if name is None:
            if source is not None:
                name = source.__name__
            else:
                name = 'element_%s' % id(self)
        self.__name__ = name

        self.__columns__ = columns
        self.__is_client__ = is_client
        for name, value in kw.items():
            if isinstance(value, str):
                value = getcolumn(self, value)
            elif hasattr(value, '__call__'):
                value = value(self)
                if isinstance(value, list):
                    value = Atom.fromlist(value)
                value = value
            if isinstance(value, list):
                value = Atom.fromlist(value)
            elif isinstance(value, np.ndarray) and not isinstance(value, Atom):
                value = Atom(value)
            setattr(self, name, value)

        if cachedir is not None:
            for name in self.__cnames__:
                if not Atom.is_cached(cachedir, name):
                    getattr(self, name).persist(cachedir, name)

    def __getattr__(self, name):
        attr = None
        if self.__cachedir__ and Atom.is_cached(self.__cachedir__, name):
            attr = Atom.load(self.__cachedir__, name, is_client=self.__is_client__)
            setattr(self, name, attr)
            return attr
        if self.__source__ is None and self.__factory__ is not None:
            source = self.__factory__()
            self.__source__ = source
            #Not really the right place for this.
            #if self.__cachedir__ is not None:
            #    print '%s = %s' % (self.__cachedir__, len(self))
        if self.__source__ is not None:
            attr = getattr(self.__source__, name)
        if attr is not None and self.__index__ is not None:
            #Although this is simple and effective it is not very efficienct
            #    a = Ai(Bi(C[xy])))
            #Using this method:
            #    ax = Cx[Bi][Ai], xy = ay[Bi][Ai]
            #A better approach is to calculate the index then apply it
            #    bai = Bi[Ai], ax = Cx[bai], ay = Cx[bai]
            attr = attr[self.__index__]
        if attr is not None:
            setattr(self, name, attr)
        return attr
    def __getitem__(self, index):
        if getattr(index, '__call__', None):
            index = index(self)
            if isinstance(index, list):
                index = Atom(index)
        return Element(self, index=index)

    def __iter__(self):
        return iter(self[i] for i in range(len(self)))

    def __len__(self):
        if self.__cnames__ is None:
            return 0
        c0 = getcolumn(self, self.__cnames__[0])
        if np.isscalar(c0):
            return 1
        else:
            return 0 if c0 is None else c0.__len__()

    def __call__(self, cnames=None, **kw):
        if cnames and hasattr(cnames, '__call__'):
            return cnames(self)
        return Element(source=self, cnames=cnames, kw=kw)

    def __getnewargs__(self):
        return ()
    def __getstate__(self):
        from collections import OrderedDict
        state = OrderedDict()
        def walk(x, stack):
            path = '.'.join(stack)
            if isinstance(x, Atom):
                state['%s' % path] = (
                    x.asarray(),
                    x.mask,
                    [(bincount.asarray(), bincount.mask) for i, bincount in enumerate(x.bincounts)])
            elif isinstance(x, Element):
                state['%s.__name__' % path] = x.__name__
                state['%s.__cnames__' % path] = x.__cnames__
                for name in x.__cnames__:
                    walk(getattr(x, name), stack + [name])
        walk(self, [self.__name__ if self.__name__ is not None else 'element_%s' % id(self)])
        return state
    def __setstate__(self, state):
        self.__source__ = None,
        self.__factory__ = None,
        self.__cachedir__ = None,
        self.__index__ = None,
        self.__cnames__ = None,
        self.__name__ = None,
        self.__columns__ = None,
        self.__is_client__ = False
        
        def toatom(x):
            return Atom(x[0], mask=x[1], bincounts=[Atom(y[0], mask=y[1]) for y in x[2]])
        class tree(object):
            def __init__(this, name=None, parent=None):
                this.name = name
                this.children = []
                if parent is not None:
                    parent.children.append(this)
            def __getattr__(this, name):
                attr = tree(name, this)
                setattr(this, name, attr)
                return attr
            def element(this, element):
                for name in this.__cnames__:
                    child = getattr(this, name)
                    if isinstance(child, tree):
                        setattr(element, name, child.element(Element(name=name)))
                    else:
                        setattr(element, name, child)
                element.__name__ = this.__name__
                element.__cnames__ = this.__cnames__
                return element
        root = tree()
        for key in state.keys():
            path = key.split('.')
            node = root
            for name in path[:-1]:
                node = getattr(node, name)
            setattr(node, path[-1],
                    toatom(state[key])
                    if isinstance(state[key], tuple) else
                    state[key])
        root.children[0].element(self)
        self.__name__ = root.children[0].__name__

    @staticmethod
    def from_pandas(df):
        '''
        '''
        return Element(
            cnames=[x for x in df.columns],
            columns=[Atom(df[name], mask=df[name].notnull())
                     for name in df.columns])

    @staticmethod
    def read_csv(*a, **kw):
        import pandas as pd
        return Element.from_pandas(pd.read_csv(*a, **kw))

    join = __join__ = join
    inner = __inner__ = inner
    left_outer = __left_outer__ = left_outer
    right_outer = __right_outer__ = right_outer
    outer = __outer__ = outer
    expand = __expand__ = expand
    demean = __demean__ = demean
    distribution = __distribution__ = distribution
    vstack = __vstack__ = vstack
    freeze = __freeze__ = freeze
    keys = __keys__ = keys
    unique = __unique__ = unique
    first = __first__ = first
    last = __last__ = last
    arange = __arange__ = arange
    zeros = __zeros__ = zeros
    ones = __ones__ = ones
    toxml = __toxml__ = toxml
    tojson = __tojson__ = tojson
    fromxml = __fromxml__ = fromxml
    to_pandas = __to_pandas__ = to_pandas
    to_csv = __to_csv__ = to_csv
    set_name = __set_name__ = set_name
    attrs = __attrs__ = attrs
    to_string = __to_string__ = to_string
    by = __by__ = by
    group_index = __group_index__ = group_index
    sort_index = __sort_index__ = sort_index
    sort_by = __sort_by__ = sort_by
    sort_index = __sort_index__ = sort_index
    sort_by = __sort_by__ = sort_by
    group = __group__ = group
    display = __display__ = display
    repeat = __repeat__ = repeat
