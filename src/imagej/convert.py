"""
Utility functions for converting between types.
"""
import ctypes
import logging
from typing import Dict

import imglyb
import numpy as np
import scyjava as sj
import xarray as xr
from jpype import JByte, JFloat, JLong, JShort

import imagej.dims as dims
import imagej.images as images
from imagej._java import jc
from imagej._java import log_exception as _log_exception

_logger = logging.getLogger(__name__)


###############
# Java images #
###############


def java_to_dataset(ij: "jc.ImageJ", data) -> "jc.Dataset":
    """
    Convert the data into an ImageJ2 Dataset.
    """
    assert sj.isjava(data)
    if isinstance(data, jc.Dataset):
        return data

    # NB: This try checking is necessary because the set of ImageJ2 converters is
    # not complete. E.g., there is no way to directly go from Img to Dataset,
    # instead you need to chain the Img->ImgPlus->Dataset converters.
    try:
        if ij.convert().supports(data, jc.Dataset):
            return ij.convert().convert(data, jc.Dataset)
        if ij.convert().supports(data, jc.ImgPlus):
            imgplus = ij.convert().convert(data, jc.ImgPlus)
            return ij.dataset().create(imgplus)
        if ij.convert().supports(data, jc.Img):
            img = ij.convert().convert(data, jc.Img)
            return ij.dataset().create(jc.ImgPlus(img))
        if ij.convert().supports(data, jc.RandomAccessibleInterval):
            rai = ij.convert().convert(data, jc.RandomAccessibleInterval)
            return ij.dataset().create(rai)
    except Exception as exc:
        _log_exception(_logger, exc)
        raise exc
    raise TypeError("Cannot convert to dataset: " + str(type(data)))


def java_to_img(ij: "jc.ImageJ", jobj) -> "jc.Img":
    """
    Convert the data into an ImgLib2 Img.
    """
    assert sj.isjava(jobj)
    if isinstance(jobj, jc.Img):
        return jobj

    # NB: This try checking is necessary because the set of ImageJ2
    # converters is not complete.
    try:
        if ij.convert().supports(jobj, jc.Img):
            return ij.convert().convert(jobj, jc.Img)
        if ij.convert().supports(jobj, jc.RandomAccessibleInterval):
            rai = ij.convert().convert(jobj, jc.RandomAccessibleInterval)
            return jc.ImgView.wrap(rai)
    except Exception as exc:
        _log_exception(_logger, exc)
        raise exc
    raise TypeError("Cannot convert to img: " + str(type(jobj)))


def imageplus_to_imgplus(ij: "jc.ImageJ", imp: "jc.ImagePlus") -> "jc.ImgPlus":
    if not jc.ImagePlus or not isinstance(imp, jc.ImagePlus):
        raise ValueError("Input is not an ImagePlus")

    ds = ij.convert().convert(imp, jc.Dataset)
    return ds.getImgPlus()


##############################
# Java image <-> NumPy array #
##############################


def ndarray_to_dataset(ij: "jc.ImageJ", narr):
    assert images.is_arraylike(narr)
    rai = imglyb.to_imglib(narr)
    return java_to_dataset(ij, rai)


def ndarray_to_img(ij: "jc.ImageJ", narr):
    assert images.is_arraylike(narr)
    rai = imglyb.to_imglib(narr)
    return java_to_img(ij, rai)


def xarray_to_dataset(ij: "jc.ImageJ", xarr):
    """
    Converts a xarray dataarray to a dataset, inverting C-style (slow axis first)
    to F-style (slow-axis last)

    :param xarr: Pass an xarray dataarray and turn into a dataset.
    :return: The dataset
    """
    assert images.is_xarraylike(xarr)
    if dims._ends_with_channel_axis(xarr):
        vals = np.moveaxis(xarr.values, -1, 0)
        dataset = ndarray_to_dataset(ij, vals)
    else:
        dataset = ndarray_to_dataset(ij, xarr.values)
    axes = dims._assign_axes(xarr)
    dataset.setAxes(axes)
    _assign_dataset_metadata(dataset, xarr.attrs)

    return dataset


def xarray_to_img(ij: "jc.ImageJ", xarr):
    """
    Converts a xarray dataarray to an ImgLib2 Img,
    inverting C-style (slow axis first) to F-style (slow axis last).

    :param xarr: Pass an xarray dataarray and turn into a img.
    :return: The img
    """
    assert images.is_xarraylike(xarr)
    if dims._ends_with_channel_axis(xarr):
        vals = np.moveaxis(xarr.values, -1, 0)
        return ndarray_to_img(ij, vals)
    else:
        return ndarray_to_img(ij, xarr.values)


def java_to_ndarray(ij: "jc.ImageJ", jobj) -> np.ndarray:
    assert sj.isjava(jobj)
    rai = ij.convert().convert(jobj, jc.RandomAccessibleInterval)
    narr = images.create_ndarray(rai)
    images.copy_rai_into_ndarray(ij, rai, narr)
    return narr


def java_to_xarray(ij: "jc.ImageJ", jobj) -> xr.DataArray:
    """
    Converts a Java image to an xarray DataArray,
    inverting F-style (slow axis last) to C-style (slow axis first).

    Labeled dimensional axes are permuted as needed
    to conform to the scikit-image standard order; see:
    https://scikit-image.org/docs/dev/user_guide/numpy_images#coordinate-conventions

    :param jobj: A RandomAccessibleInterval with axes (e.g. Dataset or ImgPlus).
    :return: xarray.DataArray with metadata/axes.
    """
    imgplus = ij.convert().convert(jobj, jc.ImgPlus)

    # Permute Java image dimensions to the scikit-image standard order.
    permuted_rai = _permute_rai_to_python(imgplus)

    # Create a new ndarray, and copy the Java image into it.
    narr = images.create_ndarray(permuted_rai)
    images.copy_rai_into_ndarray(ij, permuted_rai, narr)

    # Wrap ndarray into an xarray with axes matching the permuted RAI.
    assert hasattr(permuted_rai, "dim_axes")
    xr_axes = list(permuted_rai.dim_axes)
    xr_dims = list(permuted_rai.dims)
    xr_attrs = sj.to_python(permuted_rai.getProperties())
    # reverse axes and dims to match narr
    xr_axes.reverse()
    xr_dims.reverse()
    xr_dims = dims._convert_dims(xr_dims, direction="python")
    xr_coords = dims._get_axes_coords(xr_axes, xr_dims, narr.shape)
    return xr.DataArray(narr, dims=xr_dims, coords=xr_coords, attrs=xr_attrs)


# TODO:
# * supports_ndarray_to_dataset
# * supports_ndarray_to_img
# * supports_xarray_to_dataset
# * supports_xarray_to_img


def supports_java_to_ndarray(ij: "jc.ImageJ", obj) -> bool:
    """Return True iff conversion to ndarray is possible."""
    try:
        return ij.convert().supports(obj, jc.RandomAccessibleInterval)
    except Exception:
        return False


def supports_java_to_xarray(ij: "jc.ImageJ", obj) -> bool:
    """Return True iff conversion to ImgPlus is possible."""
    try:
        can_convert = ij.convert().supports(obj, jc.ImgPlus)
        has_axis = dims._has_axis(obj)
        return can_convert and has_axis
    except Exception:
        return False


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
    jtype_raw = sj.to_java(obj.value)
    # Then, find the correct RealType
    realtype_fqcn = _ctype_map[type(obj)]
    # jtype_raw is usually an Integer or Double.
    # We may have to cast it to fit the RealType parameter
    if realtype_fqcn in _realtype_casters:
        caster = _realtype_casters[realtype_fqcn]
        jtype_raw = caster(jtype_raw)
    # Create and return the RealType
    realtype_class = sj.jimport(realtype_fqcn)
    return realtype_class(jtype_raw)


def realtype_to_ctype(realtype):
    # First, convert the RealType to a Java primitive
    jtype_raw = realtype.get()
    # Then, convert to the python primitive
    converted = sj.to_python(jtype_raw)
    value = realtype.getClass().getName()
    for k, v in _ctype_map.items():
        if v == value:
            return k(converted)
    raise ValueError(f"Cannot convert RealType {value}")


def supports_ctype_to_realtype(obj: ctypes._SimpleCData):
    return type(obj) in _ctype_map


def supports_realtype_to_ctype(obj):
    if not isinstance(obj, sj.jimport("net.imglib2.type.numeric.RealType")):
        return False
    fqcn = obj.getClass().getName()
    return fqcn in _ctype_map.values()


####################
# Helper functions #
####################


def _assign_dataset_metadata(dataset: "jc.Dataset", attrs):
    """
    :param dataset: ImageJ2 Dataset
    :param attrs: Dictionary containing metadata
    """
    dataset.getProperties().putAll(sj.to_java(attrs))


def _permute_rai_to_python(rich_rai: "jc.RandomAccessibleInterval"):
    """Permute a RandomAccessibleInterval to the python reference order.

    Permute a RandomAccessibleInterval to the Python reference order of
    CXYZT (where dimensions exist). Note that this is reverse from the
    final array order of TZYXC.

    :param rich_rai: A RandomAccessibleInterval with axis labels
        (e.g. Dataset or ImgPlus).
    :return: A permuted RandomAccessibleInterval.
    """
    # get input rai metadata if it exists
    try:
        rai_metadata = rich_rai.getProperties()
    except AttributeError:
        rai_metadata = None

    axis_types = [axis.type() for axis in rich_rai.dim_axes]

    # permute rai to specified order and transfer metadata
    permute_order = dims.prioritize_rai_axes_order(
        axis_types, dims._python_rai_ref_order()
    )
    permuted_rai = dims.reorganize(rich_rai, permute_order)

    # add metadata to image if it exists
    if rai_metadata is not None:
        permuted_rai.getProperties().putAll(rai_metadata)

    return permuted_rai
