# General-purpose utility methods.

import jnius

# -- Python to Java --

import collections

# Adapted from code posted by vslotman on GitHub:
# https://github.com/kivy/pyjnius/issues/217#issue-145981070

jDouble        = jnius.autoclass('java.lang.Double')
jFloat         = jnius.autoclass('java.lang.Float')
jInteger       = jnius.autoclass('java.lang.Integer')
jLong          = jnius.autoclass('java.lang.Long')
jString        = jnius.autoclass('java.lang.String')
jBigDecimal    = jnius.autoclass('java.math.BigDecimal')
jBigInteger    = jnius.autoclass('java.math.BigInteger')
jArrayList     = jnius.autoclass('java.util.ArrayList')
jLinkedHashMap = jnius.autoclass('java.util.LinkedHashMap')
jLinkedHashSet = jnius.autoclass('java.util.LinkedHashSet')

class JavaNumber(object):
    '''
    Convert int/float to their corresponding Java-types based on size
    '''
    def __call__(self, obj):
        if isinstance(obj, int):
            if obj <= jInteger.MAX_VALUE:
                return jInteger(obj)
            elif obj <= jLong.MAX_VALUE:
                return jLong(obj)
            else:
                return jBigInteger(str(obj))
        elif isinstance(obj, float):
            if obj <= jFloat.MAX_VALUE:
                return jFloat(obj)
            elif obj <= jDouble.MAX_VALUE:
                return jDouble(obj)
            else:
                return jBigDecimal(str(obj))

def is_java(data):
    return isinstance(data, jnius.JavaClass) or isinstance(data, jnius.MetaJavaClass)

def to_java(data):
    '''
    Recursively convert Python object to Java object
    :param data:
    '''
    if is_java(data):
        return data

    java_type_map = {
        int:   JavaNumber(),
        str:   jString,
        float: JavaNumber()
    }
    if type(data) in java_type_map:
        # We know of a way to convert type.
        return java_type_map[type(data)](data)

    if isinstance(data, collections.Mapping):
        # Object is dict-like.
        jmap = jLinkedHashMap()
        for k, v in data.items():
            jk = to_java(k)
            jv = to_java(v)
            jmap.put(jk, jv)
        return jmap

    if isinstance(data, collections.Set):
        # Object is set-like.
        jset = jLinkedHashSet()
        for item in data:
            jitem = to_java(item)
            jset.add(jitem)
        return jset

    if isinstance(data, collections.Iterable):
        # Object is list-like.
        jlist = jArrayList()
        for item in data:
            jitem = to_java(item)
            jlist.add(jitem)
        return jlist

    raise TypeError('Unsupported type: ' + str(type(data)))

# -- Java to Python --

jObjectClass = jnius.find_javaclass('java.lang.Object')
jIterableClass = jnius.find_javaclass('java.lang.Iterable')
jCollectionClass = jnius.find_javaclass('java.util.Collection')
jIteratorClass = jnius.find_javaclass('java.util.Iterator')
jListClass = jnius.find_javaclass('java.util.List')
jMapClass = jnius.find_javaclass('java.util.Map')
jSetClass = jnius.find_javaclass('java.util.Set')

def check_instance(jclass, jobj):
    if not jclass.isInstance(jobj):
        raise TypeError('Not a ' + jclass.getName() + ': ' + jobj.getClass().getName())

def jstr(data):
    if not is_java(data):
        return '{!r}'.format(data)
    if jMapClass.isInstance(data):
        return '{' + ', '.join(jstr(k) + ': ' + jstr(v) for k,v in JavaMap(data).items()) + '}'
    if jSetClass.isInstance(data):
        return '{' + ', '.join(jstr(v) for v in JavaSet(data)) + '}'
    if jListClass.isInstance(data):
        return '[' + ', '.join(jstr(v) for v in JavaList(data)) + ']'
    if jCollectionClass.isInstance(data):
        return '[' + ', '.join(jstr(v) for v in JavaCollection(data)) + ']'
    if jIterableClass.isInstance(data):
        return '[' + ', '.join(jstr(v) for v in JavaIterable(data)) + ']'
    if jIteratorClass.isInstance(data):
        return '[' + ', '.join(jstr(v) for v in JavaIterator(data)) + ']'
    return data.toString()

class JavaObject():
    def __init__(self, jobj, jclass=jObjectClass):
        check_instance(jclass, jobj)
        self.jobj = jobj
    def __str__(self):
        return jstr(self.jobj)

class JavaIterable(JavaObject, collections.Iterable):
    def __init__(self, jobj):
        JavaObject.__init__(self, jobj, jIterableClass)
    def __iter__(self):
        return to_python(self.jobj.iterator())

class JavaCollection(JavaIterable, collections.Collection):
    def __init__(self, jobj):
        JavaObject.__init__(self, jobj, jCollectionClass)
    def __contains__(self, item):
        return to_python(self.jobj.contains(to_java(item)))
    def __len__(self):
        return to_python(self.jobj.size())
    def __eq__(self, other):
        try:
            if len(self) != len(other):
                return False
            for e1, e2 in zip(self, other):
                if e1 != e2:
                    return False
            return True
        except TypeError:
            return False

class JavaIterator(JavaObject, collections.Iterable):
    def __init__(self, jobj):
        JavaObject.__init__(self, jobj, jIteratorClass)
    def __iter__(self):
        return self
    def __next__(self):
        if self.jobj.hasNext():
            return to_python(self.jobj.next())
        raise StopIteration

class JavaList(JavaCollection, collections.MutableSequence):
    def __init__(self, jobj):
        JavaObject.__init__(self, jobj, jListClass)
    def __getitem__(self, key):
        return to_python(self.jobj.get(key))
    def __setitem__(self, key, value):
        return to_python(self.jobj.set(key, value))
    def __delitem__(self, key):
        return to_python(self.jobj.remove(key))        
    def insert(self, index, object):
        return to_python(self.jobj.set(index, object))

class JavaMap(JavaObject, collections.MutableMapping):
    def __init__(self, jobj):
        JavaObject.__init__(self, jobj, jMapClass)
    def __getitem__(self, key):
        return to_python(self.jobj.get(to_java(key)))
    def __setitem__(self, key, value):
        return to_python(self.jobj.put(to_java(key), to_java(value)))
    def __delitem__(self, key):
        return to_python(self.jobj.remove(to_python(key)))
    def keys(self):
        return to_python(self.jobj.keySet())
    def __iter__(self):
        return self.keys().__iter__()
    def __len__(self):
        return to_python(self.jobj.size())
    def __eq__(self, other):
        try:
            if len(self) != len(other):
                return False
            for k in self:
                if not k in other or self[k] != other[k]:
                    return False
            return True
        except TypeError:
            return False

class JavaSet(JavaCollection, collections.MutableSet):
    def __init__(self, jobj):
        JavaObject.__init__(self, jobj, jSetClass)
    def add(self, item):
        return to_python(self.jobj.add(to_java(item)))
    def discard(self, item):
        return to_python(self.jobj.remove(to_java(item)))
    def __iter__(self):
        return to_python(self.jobj.iterator())
    def __eq__(self, other):
        try:
            if len(self) != len(other):
                return False
            for k in self:
                if not k in other:
                    return False
            return True
        except TypeError:
            return False

def to_python(data):
    if not is_java(data):
        # Not a Java object.
        return data
    if jListClass.isInstance(data):
        return JavaList(data)
    if jMapClass.isInstance(data):
        return JavaMap(data)
    if jSetClass.isInstance(data):
        return JavaSet(data)
    if jCollectionClass.isInstance(data):
        return JavaCollection(data)
    if jIterableClass.isInstance(data):
        return JavaIterable(data)
    if jIteratorClass.isInstance(data):
        return JavaIterator(data)
    raise TypeError('Unsupported data type: ' + str(type(data)))
