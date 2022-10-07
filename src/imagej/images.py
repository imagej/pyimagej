"""
Utility functions for creating and working with images.
"""
import logging

import numpy as np
import scyjava as sj

from imagej._java import jc

_logger = logging.getLogger(__name__)


# fmt: off
_imglib2_types = {
    # "net.imglib2.type.logic.BitType":                               "bool",
    "net.imglib2.type.numeric.integer.ByteType":                      "int8",
    "net.imglib2.type.numeric.integer.ByteLongAccessType":            "int8",
    "net.imglib2.type.numeric.integer.ShortType":                     "int16",
    "net.imglib2.type.numeric.integer.ShortLongAccessType":           "int16",
    "net.imglib2.type.numeric.integer.IntType":                       "int32",
    "net.imglib2.type.numeric.integer.IntLongAccessType":             "int32",
    "net.imglib2.type.numeric.integer.LongType":                      "int64",
    "net.imglib2.type.numeric.integer.LongLongAccessType":            "int64",
    "net.imglib2.type.numeric.integer.UnsignedByteType":              "uint8",
    "net.imglib2.type.numeric.integer.UnsignedByteLongAccessType":    "uint8",
    "net.imglib2.type.numeric.integer.UnsignedShortType":             "uint16",
    "net.imglib2.type.numeric.integer.UnsignedShortLongAccessType":   "uint16",
    "net.imglib2.type.numeric.integer.UnsignedIntType":               "uint32",
    "net.imglib2.type.numeric.integer.UnsignedIntLongAccessType":     "uint32",
    "net.imglib2.type.numeric.integer.UnsignedLongType":              "uint64",
    "net.imglib2.type.numeric.integer.UnsignedLongLongAccessType":    "uint64",
    # "net.imglib2.type.numeric.ARGBType":                            "argb",
    # "net.imglib2.type.numeric.ARGBLongAccessType":                  "argb",
    "net.imglib2.type.numeric.real.FloatType":                        "float32",
    "net.imglib2.type.numeric.real.FloatLongAccessType":              "float32",
    "net.imglib2.type.numeric.real.DoubleType":                       "float64",
    "net.imglib2.type.numeric.real.DoubleLongAccessType":             "float64",
    # "net.imglib2.type.numeric.complex.ComplexFloatType":            "cfloat32",
    # "net.imglib2.type.numeric.complex.ComplexFloatLongAccessType":  "cfloat32",
    # "net.imglib2.type.numeric.complex.ComplexDoubleType":           "cfloat64",
    # "net.imglib2.type.numeric.complex.ComplexDoubleLongAccessType": "cfloat64",
}
# fmt: on


def is_arraylike(arr):
    return (
        hasattr(arr, "shape")
        and hasattr(arr, "dtype")
        and hasattr(arr, "__array__")
        and hasattr(arr, "ndim")
    )


def is_memoryarraylike(arr):
    return (
        is_arraylike(arr)
        and hasattr(arr, "data")
        and type(arr.data).__name__ == "memoryview"
    )


def is_xarraylike(xarr):
    return (
        hasattr(xarr, "values")
        and hasattr(xarr, "dims")
        and hasattr(xarr, "coords")
        and is_arraylike(xarr.values)
    )


def create_ndarray(image) -> np.ndarray:
    try:
        dtype_to_use = dtype(image)
    except TypeError:
        dtype_to_use = np.dtype("float64")

    # get shape of image and invert
    shape = list(image.shape)

    # reverse shape if image is a RandomAccessibleInterval
    if isinstance(image, jc.RandomAccessibleInterval):
        shape.reverse()

    return np.zeros(shape, dtype=dtype_to_use)


def copy_rai_into_ndarray(
    ij: "jc.ImageJ", rai: "jc.RandomAccessibleInterval", narr: np.ndarray
) -> None:
    """Copy a RandomAccessibleInterval into a numpy array.

    The input RandomAccessibleInterval is copied into the pre-initialized numpy
    array with either "fast copy" via 'net.imglib2.util.ImgUtil.copy' if
    available or the slower "copy.rai" method. Note that the input
    RandomAccessibleInterval and numpy array must have reversed dimensions
    relative to each other (e.g. [t, z, y, x, c] and [c, x, y, z, t]). Use
    _permute_rai_to_python() on the RandomAccessibleInterval to reorganize the
    dimensions.

    :param rai: A net.imglib2.RandomAccessibleInterval.
    :param narr: A NumPy array with the same shape as the input
        RandomAccessibleInterval.
    """
    if not isinstance(rai, jc.RandomAccessibleInterval):
        raise TypeError("rai is not a RAI")
    if not is_arraylike(narr):
        raise TypeError("narr is not arraylike")

    # Check imglib2 version for fast copy availability.
    imglib2_version = sj.get_version(jc.RandomAccessibleInterval)
    # TODO: After scyjava 1.6.0 is released, use:
    # sj.is_version_at_least(imglib2_version, "5.9.0")
    min_imglib2_version = "5.9.0"
    if (
        sj.compare_version(min_imglib2_version, imglib2_version)
        or imglib2_version == min_imglib2_version
    ):
        # ImgLib2 is new enough to use net.imglib2.util.ImgUtil.copy.
        ImgUtil = sj.jimport("net.imglib2.util.ImgUtil")
        ImgUtil.copy(rai, sj.to_java(narr))
        return narr

    # Check imagej-common version for fast copy availability.
    imagej_common_version = sj.get_version(jc.Dataset)
    # TODO: After scyjava 1.6.0 is released, use:
    # sj.is_version_at_least(imagej_common_version, "0.30.0")
    min_imagej_common_version = "0.30.0"
    if (
        sj.compare_version(min_imagej_common_version, imagej_common_version)
        or imagej_common_version == min_imagej_common_version
    ):
        # ImageJ Common is new enough to use (deprecated)
        # net.imagej.util.Images.copy.
        Images = sj.jimport("net.imagej.util.Images")
        Images.copy(rai, sj.to_java(narr))
        return narr

    # Fall back to copying with ImageJ Ops's copy.rai op. In theory, Ops
    # should always be faster. But in practice, the copy.rai operation is
    # slower than the hardcoded ones above. If we were to fix Ops to be
    # fast always, we could eliminate the above special casing.
    ij.op().run("copy.rai", sj.to_java(narr), rai)


def dtype(image_or_type):
    """Get the dtype of the input image as a numpy.dtype object.

    Note: for Java-based images, this is different than the image's dtype
    property, because ImgLib2-based images report their dtype as a subclass
    of net.imglib2.type.Type, and ImagePlus images do not yet implement
    the dtype function (see https://github.com/imagej/pyimagej/issues/194).

    :param image_or_type:
        | A NumPy array.
        | OR A NumPy array dtype.
        | OR An ImgLib2 image ('net.imglib2.Interval').
        | OR An ImageJ2 Dataset ('net.imagej.Dataset').
        | OR An ImageJ ImagePlus ('ij.ImagePlus').

    :return: Input image dtype.
    """
    if isinstance(image_or_type, np.dtype):
        return image_or_type
    if is_arraylike(image_or_type):
        return image_or_type.dtype
    if not sj.isjava(image_or_type):
        raise TypeError("Unsupported type: " + str(type(image_or_type)))

    # -- ImgLib2 types --
    if isinstance(image_or_type, sj.jimport("net.imglib2.type.Type")):
        for c in _imglib2_types:
            if isinstance(image_or_type, sj.jimport(c)):
                return np.dtype(_imglib2_types[c])
        raise TypeError(f"Unsupported ImgLib2 type: {image_or_type}")

    # -- ImgLib2 images --
    if isinstance(image_or_type, sj.jimport("net.imglib2.IterableInterval")):
        imglib2_type = image_or_type.firstElement()
        return dtype(imglib2_type)
    if isinstance(image_or_type, jc.RandomAccessibleInterval):
        imglib2_type = jc.Util.getTypeFromInterval(image_or_type)
        return dtype(imglib2_type)

    # -- Original ImageJ images --
    if jc.ImagePlus and isinstance(image_or_type, jc.ImagePlus):
        ij1_type = image_or_type.getType()
        ij1_types = {
            jc.ImagePlus.GRAY8: "uint8",
            jc.ImagePlus.GRAY16: "uint16",
            # NB: ImageJ's 32-bit type is float32, not uint32.
            jc.ImagePlus.GRAY32: "float32",
        }
        for t in ij1_types:
            if ij1_type == t:
                return np.dtype(ij1_types[t])
        raise TypeError(f"Unsupported original ImageJ type: {ij1_type}")

    raise TypeError("Unsupported Java type: " + str(sj.jclass(image_or_type).getName()))
