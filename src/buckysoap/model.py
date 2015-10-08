import os
import re
import sys
import inspect

def parent_iterator(link, func=lambda x: x, parent=lambda x: x.parent):
    while link is not None:
        yield func(link)
        link = parent(link)

class MetaNode(type):
    __ordinal__ = 0
    def __new__(cls, name, bases, attrs):
        new = super(MetaNode, cls).__new__(cls, name, bases, attrs)
        new.__ordinal__ = MetaNode.__ordinal__
        MetaNode.__ordinal__+=1
        return new

class Node(object):
    __metaclass__ = MetaNode

class Model(object):
    def __init__(self, root):
        self.parent = None
        self.root_distance = -1
        self.leaf_distance = 0
        self.root = Model.Class(self, root)
    def walk(self, *handlers):
        def walk(node):
            for handler in handlers:
                if handler.test(node):
                    handler(node)
            for child in node.children:
                walk(child)
        walk(self.root)
    class Node(object):
        def __init__(self, parent, source):
            self.parent = parent
            self.source = source
            self.children = []
            self.path = list(parent_iterator(self))[::-1]
            self.root_distance = self.parent.root_distance + 1
            self.leaf_distance = 0
            self.output = sys.stdout
            for i, x in enumerate(self.path[::-1]):
                if i>x.leaf_distance:
                    x.leaf_distance = i
        @property
        def name(self):
            return self.source.__name__
        @property
        def doc(self):
            if self.source.__doc__ is None:
                return ''
            else:
                return re.sub('^\s+', '', self.source.__doc__).strip()
        def write(self, lines, pad=0):
            padding = (self.root_distance + pad) * '    '
            self.output.write(padding + ('\n'+padding).join(lines.split('\n')) + '\n')
    class Class(Node):
        def __init__(self, parent, source):
            super(Model.Class, self).__init__(parent, source)
            self.classes = sorted(
                [Model.Class(self, value)
                 for name, value in inspect.getmembers(self.source, inspect.isclass)
                 if not name.startswith('__') and value is not type],
                key=lambda x: x.ordinal)
            self.methods = sorted(
                [Model.Method(self, function)
                 for name, function in inspect.getmembers(self.source, inspect.ismethod)
                 if not name.startswith('__')],
                key=lambda x: x.lineno)
            self.children = self.classes + self.methods
        @property
        def instance(self):
            return self.source()
        @property
        def ordinal(self):
            return self.source.__ordinal__
    class Method(Node):
        @property
        def code(self):
            code = inspect.getsource(self.source)
            return re.sub('^\s+', '', code[code.rindex("'''")+4:], flags=re.M)
        @property
        def result(self):
            return self.source(self.parent.instance)
        @property
        def lineno(self):
            return self.source.im_func.func_code.co_firstlineno

def handler(test):
    def wrap(func):
        func.test = test
        return func
    return wrap

class Generator(object):
    def __init__(self, generated_root):
        from mako.lookup import TemplateLookup
        self.generated_root = generated_root
        self.template_lookup = TemplateLookup(directories=['templates'])
    def render(self, target, template=None, **sources):
        import os
        from mako.template import Template
        if target is not None:
            target = os.path.join(self.generated_root, Template(target).render(**sources))
            dirname = os.path.dirname(target)
            if dirname!='' and not os.path.exists(dirname):
                os.mkdir(dirname)
        def render(page):
            if target is not None:
                print "Outputing", target
            if template is not None:
                print >> page, Template(
                    '<%include file="'+template+'"/>',
                    lookup = self.template_lookup
                ).render(**sources)
        if target is None:
            render(sys.stdout)
        else:
            with open(target, 'w') as page:
                render(page)
