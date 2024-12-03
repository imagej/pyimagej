import ctypes

import pytest
import scyjava as sj

parameters = [
    (ctypes.c_bool, "net.imglib2.type.logic.BoolType", True),
    (ctypes.c_byte, "net.imglib2.type.numeric.integer.ByteType", 4),
    (ctypes.c_ubyte, "net.imglib2.type.numeric.integer.UnsignedByteType", 4),
    (ctypes.c_int8, "net.imglib2.type.numeric.integer.ByteType", 4),
    (ctypes.c_uint8, "net.imglib2.type.numeric.integer.UnsignedByteType", 4),
    (ctypes.c_short, "net.imglib2.type.numeric.integer.ShortType", 4),
    (ctypes.c_ushort, "net.imglib2.type.numeric.integer.UnsignedShortType", 4),
    (ctypes.c_int16, "net.imglib2.type.numeric.integer.ShortType", 4),
    (ctypes.c_uint16, "net.imglib2.type.numeric.integer.UnsignedShortType", 4),
    (ctypes.c_int32, "net.imglib2.type.numeric.integer.IntType", 4),
    (ctypes.c_uint32, "net.imglib2.type.numeric.integer.UnsignedIntType", 4),
    (ctypes.c_uint64, "net.imglib2.type.numeric.integer.UnsignedLongType", 4),
    (ctypes.c_int64, "net.imglib2.type.numeric.integer.LongType", 4),
    (ctypes.c_uint64, "net.imglib2.type.numeric.integer.UnsignedLongType", 4),
    (ctypes.c_longlong, "net.imglib2.type.numeric.integer.LongType", 4),
    (ctypes.c_ulonglong, "net.imglib2.type.numeric.integer.UnsignedLongType", 4),
    (ctypes.c_float, "net.imglib2.type.numeric.real.FloatType", 4.5),
    (ctypes.c_double, "net.imglib2.type.numeric.real.DoubleType", 4.5),
]


# -- Tests --


@pytest.mark.parametrize(argnames="ctype,jtype_str,value", argvalues=parameters)
def test_ctype_to_realtype(ij, ctype, jtype_str, value):
    py_type = ctype(value)
    # Convert the ctype into a RealType
    converted = ij.py.to_java(py_type)
    jtype = sj.jimport(jtype_str)
    assert isinstance(converted, jtype)
    assert converted.get() == value
    # Convert the RealType back into a ctype
    converted_back = ij.py.from_java(converted)
    assert isinstance(converted_back, ctype)
    assert converted_back.value == value
