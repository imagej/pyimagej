"""
Utility functions for converting between types.
"""
import ctypes
from typing import Dict

from jpype import JByte, JFloat, JLong, JShort
from scyjava import jimport, to_java, to_python

######################
# ctype <-> RealType #
######################

# Dict between ctypes and equivalent RealTypes
# These types were chosen to guarantee the number of bits in each
# ctype, as the sizes of some ctypes are platform-dependent. See
# https://docs.python.org/3/library/ctypes.html#ctypes-fundamental-data-types-2
# for more information.
_ctype_map: Dict[type, str] = {
    ctypes.c_bool: "net.imglib2.type.logic.BoolType",
    ctypes.c_int8: "net.imglib2.type.numeric.integer.ByteType",
    ctypes.c_uint8: "net.imglib2.type.numeric.integer.UnsignedByteType",
    ctypes.c_int16: "net.imglib2.type.numeric.integer.ShortType",
    ctypes.c_uint16: "net.imglib2.type.numeric.integer.UnsignedShortType",
    ctypes.c_int32: "net.imglib2.type.numeric.integer.IntType",
    ctypes.c_uint32: "net.imglib2.type.numeric.integer.UnsignedIntType",
    ctypes.c_int64: "net.imglib2.type.numeric.integer.LongType",
    ctypes.c_uint64: "net.imglib2.type.numeric.integer.UnsignedLongType",
    ctypes.c_float: "net.imglib2.type.numeric.real.FloatType",
    ctypes.c_double: "net.imglib2.type.numeric.real.DoubleType",
}

# Dict of casters for realtypes that cannot directly take
# the raw conversion of ctype.value
_realtype_casters: Dict[str, type] = {
    "net.imglib2.type.numeric.integer.ByteType": JByte,
    "net.imglib2.type.numeric.integer.UnsignedIntType": JLong,
    "net.imglib2.type.numeric.integer.ShortType": JShort,
    "net.imglib2.type.numeric.integer.LongType": JLong,
    "net.imglib2.type.numeric.integer.UnsignedLongType": JLong,
    "net.imglib2.type.numeric.real.DoubleType": JFloat,
}


def ctype_to_realtype(obj: ctypes._SimpleCData):
    # First, convert the ctype value to java
    jtype_raw = to_java(obj.value)
    # Then, find the correct RealType
    realtype_fqcn = _ctype_map[type(obj)]
    # jtype_raw is usually an Integer or Double.
    # We may have to cast it to fit the RealType parameter
    if realtype_fqcn in _realtype_casters:
        caster = _realtype_casters[realtype_fqcn]
        jtype_raw = caster(jtype_raw)
    # Create and return the RealType
    realtype_class = jimport(realtype_fqcn)
    return realtype_class(jtype_raw)


def realtype_to_ctype(realtype):
    # First, convert the RealType to a Java primitive
    jtype_raw = realtype.get()
    # Then, convert to the python primitive
    converted = to_python(jtype_raw)
    value = realtype.getClass().getName()
    for k, v in _ctype_map.items():
        if v == value:
            return k(converted)
    raise ValueError(f"Cannot convert RealType {value}")


def supports_ctype_to_realtype(obj: ctypes._SimpleCData):
    return type(obj) in _ctype_map


def supports_realtype_to_ctype(obj):
    if not isinstance(obj, jimport("net.imglib2.type.numeric.RealType")):
        return False
    fqcn = obj.getClass().getName()
    return fqcn in _ctype_map.values()
