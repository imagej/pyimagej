"""
Utility functions for querying and manipulating dimensional axis metadata.
"""
import logging
from typing import List, Tuple, Union

import numpy as np
import scyjava as sj
import xarray as xr
from jpype import JException, JObject

from imagej._java import jc
from imagej.images import is_arraylike as _is_arraylike
from imagej.images import is_xarraylike as _is_xarraylike

_logger = logging.getLogger(__name__)


def get_axes(
    rai: "jc.RandomAccessibleInterval",
) -> List["jc.CalibratedAxis"]:
    """
    imagej.dims.get_axes(image) is deprecated. Use image.dim_axes instead.
    """
    _logger.warning(
        "imagej.dims.get_axes(image) is deprecated. Use image.dim_axes instead."
    )
    return [
        (JObject(rai.axis(idx), jc.CalibratedAxis))
        for idx in range(rai.numDimensions())
    ]


def get_axis_types(rai: "jc.RandomAccessibleInterval") -> List["jc.AxisType"]:
    """
    imagej.dims.get_axis_types(image) is deprecated. Use this code instead:

        axis_types = [axis.type() for axis in image.dim_axes]
    """
    _logger.warning(
        "imagej.dims.get_axis_types(image) is deprecated. Use this code instead:\n"
        + "\n"
        + "    axis_types = [axis.type() for axis in image.dim_axes]"
    )
    if _has_axis(rai):
        rai_dims = get_dims(rai)
        for i in range(len(rai_dims)):
            if rai_dims[i].lower() == "c":
                rai_dims[i] = "Channel"
            if rai_dims[i].lower() == "t":
                rai_dims[i] = "Time"
        rai_axis_types = []
        for i in range(len(rai_dims)):
            rai_axis_types.append(jc.Axes.get(rai_dims[i]))
        return rai_axis_types
    else:
        raise AttributeError(
            f"Unsupported Java type: {type(rai)} has no axis attribute."
        )


def get_dims(image) -> List[str]:
    """
    imagej.dims.get_dims(image) is deprecated. Use image.shape and image.dims instead.
    """
    _logger.warning(
        "imagej.dims.get_dims(image) is deprecated. Use image.shape and image.dims "
        "instead."
    )
    if _is_xarraylike(image):
        return image.dims
    if _is_arraylike(image):
        return image.shape
    if hasattr(image, "axis"):
        axes = get_axes(image)
        return _get_axis_labels(axes)
    if isinstance(image, jc.RandomAccessibleInterval):
        return list(image.dimensionsAsLongArray())
    if isinstance(image, jc.ImagePlus):
        shape = image.getDimensions()
        return [axis for axis in shape if axis > 1]
    raise TypeError(f"Unsupported image type: {image}\n No dimensions or shape found.")


def get_shape(image) -> List[int]:
    """
    imagej.dims.get_shape(image) is deprecated. Use image.shape instead.
    """
    _logger.warning(
        "imagej.dims.get_shape(image) is deprecated. Use image.shape instead."
    )
    if _is_arraylike(image):
        return list(image.shape)
    if not sj.isjava(image):
        raise TypeError("Unsupported type: " + str(type(image)))
    if isinstance(image, jc.Dimensions):
        return [image.dimension(d) for d in range(image.numDimensions())]
    if isinstance(image, jc.ImagePlus):
        shape = image.getDimensions()
        return [axis for axis in shape if axis > 1]
    raise TypeError(f"Unsupported Java type: {str(sj.jclass(image).getName())}")


def reorganize(
    rai: "jc.RandomAccessibleInterval", permute_order: List[int]
) -> "jc.ImgPlus":
    """Reorganize the dimension order of a RandomAccessibleInterval.

    Permute the dimension order of an input RandomAccessibleInterval using
    a List of ints (i.e. permute_order) to determine the shape of the output ImgPlus.

    :param rai: A RandomAccessibleInterval,
    :param permute_order: List of int in which to permute the RandomAccessibleInterval.
    :return: A permuted ImgPlus.
    """
    img = _dataset_to_imgplus(rai)

    # check for dimension count mismatch
    dim_num = rai.numDimensions()

    if len(permute_order) != dim_num:
        raise ValueError(
            f"Mismatched dimension count: {len(permute_order)} != {dim_num}"
        )

    # get ImageJ resources
    ImgView = sj.jimport("net.imglib2.img.ImgView")

    # copy dimensional axes into
    axes = []
    for i in range(dim_num):
        old_dim = permute_order[i]
        axes.append(img.axis(old_dim))

    # repeatedly permute the image dimensions into shape
    rai = img.getImg()
    for i in range(dim_num):
        old_dim = permute_order[i]
        if old_dim == i:
            continue
        rai = jc.Views.permute(rai, old_dim, i)

        # update index mapping acccordingly...this is hairy ;-)
        for j in range(dim_num):
            if permute_order[j] == i:
                permute_order[j] = old_dim
                break

        permute_order[i] = i

    return jc.ImgPlus(ImgView.wrap(rai), img.getName(), axes)


def prioritize_rai_axes_order(
    axis_types: List["jc.AxisType"], ref_order: List["jc.AxisType"]
) -> List[int]:
    """Prioritize the axes order to match a reference order.

    The input List of 'AxisType' from the image to be permuted
    will be prioritized to match (where dimensions exist) to
    a reference order (e.g. _python_rai_ref_order).

    :param axis_types: List of 'net.imagej.axis.AxisType' from image.
    :param ref_order: List of 'net.imagej.axis.AxisType' from reference order.
    :return: List of int for permuting a image (e.g. [0, 4, 3, 1, 2])
    """
    permute_order = []
    for axis in ref_order:
        for i in range(len(axis_types)):
            if axis == axis_types[i]:
                permute_order.append(i)

    for i in range(len(axis_types)):
        if axis_types[i] not in ref_order:
            permute_order.append(i)

    return permute_order


def _assign_axes(
    xarr: xr.DataArray,
) -> List[Union["jc.DefaultLinearAxis", "jc.EnumeratedAxis"]]:
    """
    Obtain xarray axes names, origin, scale and convert into ImageJ Axis. Supports both
    DefaultLinearAxis and the newer EnumeratedAxis.
    :param xarr: xarray that holds the data.
    :return: A list of ImageJ Axis with the specified origin and scale.
    """
    axes = [""] * xarr.ndim
    for dim in xarr.dims:
        axis_str = _convert_dim(dim, "java")
        ax_type = jc.Axes.get(axis_str)
        ax_num = _get_axis_num(xarr, dim)
        coords_arr = xarr.coords[dim].to_numpy()

        # check if coords/scale is numeric
        if _is_numeric_scale(coords_arr):
            doub_coords = [jc.Double(np.double(x)) for x in xarr.coords[dim]]
        else:
            _logger.warning(
                f"The {ax_type.label} axis is non-numeric and is translated "
                "to a linear index."
            )
            doub_coords = [
                jc.Double(np.double(x)) for x in np.arrange(len(xarr.coords[dim]))
            ]

        # assign calibrated axis type -- checks xarray for imagej metadata
        jaxis = None
        if "imagej" in xarr.attrs.keys():
            ij_dim = _convert_dim(dim, "java")
            if ij_dim + "_cal_axis_type" in xarr.attrs["imagej"].keys():
                cal_axis_type = xarr.attrs["imagej"][ij_dim + "_cal_axis_type"]
                # get scale from metadata if axis type is DefaultLinearAxis
                if cal_axis_type == "DefaultLinearAxis":
                    origin = xarr.attrs["imagej"][ij_dim + "_origin"]
                    scale = xarr.attrs["imagej"][ij_dim + "_scale"]
                    jaxis = _str_to_cal_axis(cal_axis_type)(ax_type, scale, origin)
                else:
                    try:
                        jaxis = _str_to_cal_axis(cal_axis_type)(ax_type, doub_coords)
                    except (JException, TypeError):
                        jaxis = _get_fallback_linear_axis(ax_type, doub_coords)
            else:
                jaxis = _get_fallback_linear_axis(ax_type, doub_coords)
        else:
            jaxis = _get_fallback_linear_axis(ax_type, doub_coords)

        axes[ax_num] = jaxis

    return axes


def _ends_with_channel_axis(xarr: xr.DataArray) -> bool:
    """Check if xarray.DataArray ends in the channel dimension.
    :param xarr: xarray.DataArray to check.
    :return: Boolean
    """
    ends_with_axis = xarr.dims[len(xarr.dims) - 1].lower() in ["c", "ch", "channel"]
    return ends_with_axis


def _get_axis_num(xarr: xr.DataArray, axis):
    """
    Get the xarray -> java axis number due to inverted axis order for C style numpy
    arrays (default)

    :param xarr: Xarray to convert
    :param axis: Axis number to convert
    :return: Axis idx in java
    """
    py_axnum = xarr.get_axis_num(axis)
    if np.isfortran(xarr.values):
        return py_axnum

    if _ends_with_channel_axis(xarr):
        if axis == len(xarr.dims) - 1:
            return axis
        else:
            return len(xarr.dims) - py_axnum - 2
    else:
        return len(xarr.dims) - py_axnum - 1


def _get_axes_coords(
    axes: List["jc.CalibratedAxis"], dims: List[str], shape: Tuple[int]
) -> dict:
    """
    Get xarray style coordinate list dictionary from a dataset
    :param axes: List of ImageJ axes
    :param dims: List of axes labels for each dataset axis
    :param shape: F-style, or reversed C-style, shape of axes numpy array.
    :return: Dictionary of coordinates for each axis.
    """
    coords = {
        dims[idx]: [
            axes[idx].calibratedValue(position) for position in range(shape[idx])
        ]
        for idx in range(len(dims))
    }
    return coords


def _get_scale(axis):
    """
    Get the scale of an axis, assuming it is linear and so the scale is simply
    second - first coordinate.

    :param axis: A 1D list like entry accessible with indexing, which contains the
        axis coordinates
    :return: The scale for this axis or None if it is a non-numeric scale.
    """
    try:
        # HACK: This axis length check is a work around for singleton dimensions.
        # You can't calculate the slope of a singleton dimension.
        # This section will be removed when axis-scale-logic is merged.
        if len(axis) <= 1:
            return 1
        else:
            return axis.values[1] - axis.values[0]
    except TypeError:
        return None


def _is_numeric_scale(coords_array: np.ndarray) -> bool:
    """
    Checks if the coordinates array of the given axis is numeric.

    :param coords_array: A 1D NumPy array.
    :return: bool
    """
    return np.issubdtype(coords_array.dtype, np.number)


def _get_fallback_linear_axis(axis_type: "jc.AxisType", values):
    """
    Get a DefaultLinearAxis manually when all other axes
    resources are unavailable.
    """
    origin = values[0]
    # calculate the slope using the values/coord array
    if len(values) <= 1:
        scale = 1
    else:
        scale = values[1] - values[0]
    return jc.DefaultLinearAxis(axis_type, scale, origin)


def _dataset_to_imgplus(rai: "jc.RandomAccessibleInterval") -> "jc.ImgPlus":
    """Get an ImgPlus from a Dataset.

    Get an ImgPlus from a Dataset or just return the RandomAccessibleInterval
    if its not a Dataset.

    :param rai: A RandomAccessibleInterval.
    :return: The ImgPlus from a Dataset.
    """
    if isinstance(rai, jc.Dataset):
        return rai.getImgPlus()
    else:
        return rai


def _get_axis_labels(axes: List["jc.CalibratedAxis"]) -> List[str]:
    """Get the axes labels from a List of 'CalibratedAxis'.

    Extract the axis labels from a List of 'CalibratedAxis'.

    :param axes: A List of 'CalibratedAxis'.
    :return: A list of the axis labels.
    """
    return [str((axes[idx].type().getLabel())) for idx in range(len(axes))]


def _python_rai_ref_order() -> List["jc.AxisType"]:
    """Get the Java style numpy reference order.

    Get a List of 'AxisType' in the Python/scikitimage
    preferred order. Note that this reference order is
    reversed.
    :return: List of dimensions in numpy preferred order.
    """
    return [jc.Axes.CHANNEL, jc.Axes.X, jc.Axes.Y, jc.Axes.Z, jc.Axes.TIME]


def _convert_dim(dim: str, direction: str) -> str:
    """Convert a dimension to Python/NumPy or ImageJ convention.

    Convert a single dimension to Python/NumPy or ImageJ convention by
    indicating which direction ('python' or 'java'). A converted dimension
    is returned.

    :param dim: A dimension to be converted.
    :param direction:
        'python': Convert a single dimension from ImageJ to Python/NumPy convention.
        'java': Convert a single dimension from Python/NumPy to ImageJ convention.
    :return: A single converted dimension.
    """
    if direction.lower() == "python":
        return _to_pydim(dim)
    elif direction.lower() == "java":
        return _to_ijdim(dim)
    else:
        return dim


def _convert_dims(dimensions: List[str], direction: str) -> List[str]:
    """Convert a List of dimensions to Python/NumPy or ImageJ conventions.

    Convert a List of dimensions to Python/Numpy or ImageJ conventions by
    indicating which direction ('python' or 'java'). A List of converted
    dimentions is returned.

    :param dimensions: List of dimensions (e.g. X, Y, Channel, Z, Time)
    :param direction:
        'python': Convert dimensions from ImageJ to Python/NumPy conventions.
        'java': Convert dimensions from Python/NumPy to ImageJ conventions.
    :return: List of converted dimensions.
    """
    new_dims = []

    if direction.lower() == "python":
        for dim in dimensions:
            new_dims.append(_to_pydim(dim))
        return new_dims
    elif direction.lower() == "java":
        for dim in dimensions:
            new_dims.append(_to_ijdim(dim))
        return new_dims
    else:
        return dimensions


def _validate_dim_order(dim_order: List[str], shape: tuple) -> List[str]:
    """
    Validate a List of dimensions. If the dimension list is smaller
    fill the rest of the list with "dim_n" (following xarrray convention).

    :param dim_order: List of dimensions (e.g. X, Y, Channel, Z, Time)
    :param shape: Shape image for the dimension order.
    :return: List with "dim_n" dimensions added to match shape length.
    """
    dim_len = len(dim_order)
    shape_len = len(shape)
    if dim_len < shape_len:
        d = shape_len - dim_len
        for i in range(d):
            dim_order.append(f"dim_{i}")
        return dim_order
    if dim_len > shape_len:
        raise ValueError(f"Expected {shape_len} dimensions but got {dim_len}.")
    return dim_order


def _has_axis(rai: "jc.RandomAccessibleInterval"):
    """Check if a RandomAccessibleInterval has axes."""
    if sj.isjava(rai):
        return hasattr(rai, "axis")
    else:
        False


def _to_pydim(key: str) -> str:
    """Convert ImageJ dimension convention to Python/NumPy."""
    pydims = {
        "Time": "t",
        "slice": "pln",
        "Z": "pln",
        "Y": "row",
        "X": "col",
        "Channel": "ch",
    }

    if key in pydims:
        return pydims[key]
    else:
        return key


def _to_ijdim(key: str) -> str:
    """Convert Python/NumPy dimension convention to ImageJ."""
    ijdims = {
        "col": "X",
        "x": "X",
        "row": "Y",
        "y": "Y",
        "ch": "Channel",
        "c": "Channel",
        "pln": "Z",
        "z": "Z",
        "t": "Time",
    }

    if key in ijdims:
        return ijdims[key]
    else:
        return key


def _cal_axis_to_str(key) -> str:
    """
    Convert a CalibratedAxis class (e.g. net.imagej.axis.DefaultLinearAxis) to
    a string.
    """
    cal_axis_to_str = {
        jc.ChapmanRichardsAxis: "ChapmanRichardsAxis",
        jc.DefaultLinearAxis: "DefaultLinearAxis",
        jc.EnumeratedAxis: "EnumeratedAxis",
        jc.ExponentialAxis: "ExponentialAxis",
        jc.ExponentialRecoveryAxis: "ExponentialRecoveryAxis",
        jc.GammaVariateAxis: "GammaVariateAxis",
        jc.GaussianAxis: "GaussianAxis",
        jc.IdentityAxis: "IdentityAxis",
        jc.InverseRodbardAxis: "InverseRodbardAxis",
        jc.LogLinearAxis: "LogLinearAxis",
        jc.PolynomialAxis: "PolynomialAxis",
        jc.PowerAxis: "PowerAxis",
        jc.RodbardAxis: "RodbardAxis",
    }

    if key.__class__ in cal_axis_to_str:
        return cal_axis_to_str[key.__class__]
    else:
        return "unknown"


def _str_to_cal_axis(key: str):
    """
    Convert a string (e.g. "DefaultLinearAxis") to a CalibratedAxis class.
    """
    str_to_cal_axis = {
        "ChapmanRichardsAxis": jc.ChapmanRichardsAxis,
        "DefaultLinearAxis": jc.DefaultLinearAxis,
        "EnumeratedAxis": jc.EnumeratedAxis,
        "ExponentialAxis": jc.ExponentialAxis,
        "ExponentialRecoveryAxis": jc.ExponentialRecoveryAxis,
        "GammaVariateAxis": jc.GammaVariateAxis,
        "GaussianAxis": jc.GaussianAxis,
        "IdentityAxis": jc.IdentityAxis,
        "InverseRodbardAxis": jc.InverseRodbardAxis,
        "LogLinearAxis": jc.LogLinearAxis,
        "PolynomialAxis": jc.PolynomialAxis,
        "PowerAxis": jc.PowerAxis,
        "RodbardAxis": jc.RodbardAxis,
    }

    if key in str_to_cal_axis:
        return str_to_cal_axis[key]
    else:
        return None
