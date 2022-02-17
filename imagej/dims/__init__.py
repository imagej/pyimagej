import logging
from matplotlib.pyplot import jet
import scyjava as sj
import numpy as np
from jpype import JObject, JException
from typing import List, Tuple

def get_axes(rai: 'RandomAccessibleInterval') -> List['CalibratedAxis']:
    """Get a List of 'CalibratedAxis'.

    Get a List of 'CalibratedAxis' from a RandomAccessibleInterval. Note that
    Dataset and ImgPlus have axes. Other inervals may not have axes, such as 
    a PlanarImg.

    :param rai: Input Dataset or RandomAccessibleInterval.
    :return: A List of 'CalibratedAxis'.
    """
    return [(JObject(rai.axis(idx), sj.jimport('net.imagej.axis.CalibratedAxis'))) for idx in range(rai.numDimensions())]


def get_axis_types(rai: 'RandomAccessibleInterval') -> List['AxisType']:
    """Get a list of 'AxisType' from a RandomAccessibleInterval.

    Get a List of 'AxisType' from a RandomAccessibleInterval. Note that Dataset
    and ImgPlus have axes. Other intervals may not have axes, such as
    a PlanarImg.

    :param rai: A RandomAccessibleInterval with axes.
    :return: A List of 'AxisType'.
    """
    if _has_axis(rai):
        Axes = sj.jimport('net.imagej.axis.Axes')
        rai_dims = get_dims(rai)
        for i in range(len(rai_dims)):
            if rai_dims[i] == 'C' or rai_dims[i] == 'c':
                rai_dims[i] = 'Channel'
            if rai_dims[i] == 'T' or rai_dims[i] == 't':
                rai_dims[i] = 'Time'
        rai_axis_types = []
        for i in range(len(rai_dims)):
            rai_axis_types.append(Axes.get(rai_dims[i]))
        return rai_axis_types
    else:
        raise AttributeError(f"Unsupported Java type: {type(rai)} has no axis attribute.")


def get_dims(image) -> List[str]:
    """Get the dimensions of an image.

    Get the dimensions (e.g. TZYXC) of an image. If no dimension
    labels are found, the shape of the image is returned.

    :param image:
        An numpy or xarray.DataArray
        OR An ImgLib2 image ('net.imglib2.Interval').
        OR An ImageJ2 Dataset ('net.imagej.Dataset').
        OR An ImageJ ImagePlus ('ij.ImagePlus').
    :return: List of dimensions.
    """
    if _is_xarraylike(image):
        return image.dims
    if _is_arraylike(image):
        return image.shape
    if hasattr(image, 'axis'):
        axes = get_axes(image)
        return _get_axis_labels(axes)
    if isinstance(image, sj.jimport('net.imglib2.RandomAccessibleInterval')):
        return list(image.dimensionsAsLongArray())
    raise TypeError(f"Unsupported image type: {image}\n No dimensions or shape found.")


def get_shape(image) -> List[int]:
    """Get the shape of an image.

    Get the shape of an image.

    :param image: An image (e.g. xarray, numpy, ImagePlus)
    :return: Shape of the image.
    """
    if _is_arraylike(image):
        return list(image.shape)
    if not sj.isjava(image):
        raise TypeError('Unsupported type: ' + str(type(image)))
    if isinstance(image, sj.jimport('net.imglib2.Dimensions')):
        return [image.dimension(d) for d in range(image.numDimensions())]
    if isinstance(image, sj.jimport('ij.ImagePlus')):
        return image.getDimensions()
    raise TypeError(f'Unsupported Java type: {str(sj.jclass(image).getName())}')


def reorganize(rai: 'RandomAccessibleInterval', permute_order: List[int]) -> 'ImgPlus':
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
        raise ValueError(f"Mismatched dimension count: {len(permute_order)} != {dim_num}")

    # get ImageJ resources
    ImgPlus = sj.jimport('net.imagej.ImgPlus')
    ImgView = sj.jimport('net.imglib2.img.ImgView')
    Views = sj.jimport('net.imglib2.view.Views')

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
        rai = Views.permute(rai, old_dim, i)

        # update index mapping acccordingly...this is hairy ;-)
        for j in range(dim_num):
            if permute_order[j] == i:
                permute_order[j] = old_dim
                break

        permute_order[i] = i

    return ImgPlus(ImgView.wrap(rai), img.getName(), axes)


def prioritize_rai_axes_order(axis_types: List['AxisType'], ref_order: List['AxisType']) -> List[int]:
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
                permute_order.insert(1, i)

    return permute_order


def prioritize_xarray_axes_order(dimensions: List[str], ref_order: List[str]) -> List[str]:
    """Prioritize the axes order to match a reference order.

    The input List of dimensions (type: str) from the xarray.DataArray 
    to be transposed will be prioritizied to match (where dimensions exist)
    to a reference order (e.g. _java_numpy_ref_order).

    :param dimensions: List of dimensions (e.g. ['x', 'y', 'c'])
    :return: List of dimensions for transposing an xarray.DataArray.
    """
    transpose_order = []
    for dim in ref_order:
        for i in range(len(dimensions)):
            if dim == dimensions[i]:
                transpose_order.append(dim)

    for i in range(len(dimensions)):
        if dimensions[i] not in ref_order:
            transpose_order.insert(1, dimensions[i])

    return transpose_order

def _assign_axes(xarr):
    """
    Obtain xarray axes names, origin, and scale and convert into ImageJ Axis; currently supports EnumeratedAxis
    :param xarr: xarray that holds the units
    :return: A list of ImageJ Axis with the specified origin and scale
    """
    Axes = sj.jimport('net.imagej.axis.Axes')
    Double = sj.jimport('java.lang.Double')

    axes = ['']*len(xarr.dims)

    # try to get EnumeratedAxis, if not then default to LinearAxis in the loop
    try:
        EnumeratedAxis = _get_enumerated_axis()
    except(JException, TypeError):
        EnumeratedAxis = None

    for dim in xarr.dims:
        axis_str = _convert_dim(dim, direction='java')
        ax_type = Axes.get(axis_str)
        ax_num = _get_axis_num(xarr, dim)
        scale = _get_scale(xarr.coords[dim])

        if scale is None:
            logging.warning(f"The {ax_type.label} axis is non-numeric and is translated to a linear index.")
            doub_coords = [Double(np.double(x)) for x in np.arange(len(xarr.coords[dim]))]
        else:
            doub_coords = [Double(np.double(x)) for x in xarr.coords[dim]]

        # EnumeratedAxis is a new axis made for xarray, so is only present in ImageJ versions that are released
        # later than March 2020.  This actually returns a LinearAxis if using an earlier version.
        if EnumeratedAxis != None:
            java_axis = EnumeratedAxis(ax_type, sj.to_java(doub_coords))
        else:
            java_axis = _get_linear_axis(ax_type, sj.to_java(doub_coords))

        axes[ax_num] = java_axis

    return axes


def _ends_with_channel_axis(xarr):
    ends_with_axis = xarr.dims[len(xarr.dims)-1].lower() in ['c', 'ch','channel']
    return ends_with_axis


def _get_axis_num(xarr, axis):
    """
    Get the xarray -> java axis number due to inverted axis order for C style numpy arrays (default)
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


def _get_axes_coords(axes, dims, shape):
    """
    Get xarray style coordinate list dictionary from a dataset
    :param axes: List of ImageJ axes
    :param dims: List of axes labels for each dataset axis
    :param shape: F-style, or reversed C-style, shape of axes numpy array.
    :return: Dictionary of coordinates for each axis.
    """
    coords = {dims[idx]: [axes[idx].calibratedValue(position) for position in range(shape[idx])]
                for idx in range(len(dims))}
    return coords


def _get_scale(axis):
    """
    Get the scale of an axis, assuming it is linear and so the scale is simply second - first coordinate.
    :param axis: A 1D list like entry accessible with indexing, which contains the axis coordinates
    :return: The scale for this axis or None if it is a non-numeric scale.
    """
    try:
        return axis.values[1] - axis.values[0]
    except TypeError:
        return None

def _get_enumerated_axis():
    # EnumeratedAxis is a new axis made for xarray, so is only present in
    # ImageJ versions that are released later than March 2020. This check
    # defaults to LinearAxis instead if Enumerated does not work.
    return sj.jimport('net.imagej.axis.EnumeratedAxis')

def _get_linear_axis(axis_type: 'AxisType', values):
    DefaultLinearAxis = sj.jimport('net.imagej.axis.DefaultLinearAxis')
    origin = values[0]
    scale = values[1] - values[0]
    axis = DefaultLinearAxis(axis_type, scale, origin)
    return axis


def _dataset_to_imgplus(rai: 'RandomAccessibleInterval') -> 'ImgPlus':
    """Get an ImgPlus from a Dataset.

    Get an ImgPlus from a Dataset or just return the RandomAccessibleInterval
    if its not a Dataset.

    :param rai: A RandomAccessibleInterval.
    :return: The ImgPlus from a Dataset.
    """
    if isinstance(rai, sj.jimport('net.imagej.Dataset')):
        return rai.getImgPlus()
    else:
        return rai


def _get_axis_labels(axes: List['CalibratedAxis']) -> List[str]:
    """Get the axes labels from a List of 'CalibratedAxis'.

    Extract the axis labels from a List of 'CalibratedAxis'.

    :param axes: A List of 'CalibratedAxis'.
    :return: A list of the axis labels.
    """
    return [str((axes[idx].type().getLabel())) for idx in range(len(axes))]


def _python_rai_ref_order() -> List['AxisType']:
    """Get the Java style numpy reference order.

    Get a List of 'AxisType' in the Python/scikitimage
    preferred order. Note that this reference order is
    reversed.
    :return: List of dimensions in numpy preferred order.
    """
    Axes = sj.jimport('net.imagej.axis.Axes')

    return [Axes.CHANNEL, Axes.X, Axes.Y, Axes.Z, Axes.TIME]


def _java_numpy_ref_order() -> List[str]:
    """Get the numpy style Jav reference order.

    Get a List of str in the Java preferred order.
    Note that this reference order is reversed.
    :return: List of dimensions in Java preferred order.
    """
    return ['t', 'z', 'y', 'x', 'c']


def _convert_dim(dim: str, direction: str) -> str:
    if direction.lower() == 'python':
        return _to_pydim(dim)
    elif direction.lower() == 'java':
        return _to_ijdim(dim)
    else:
        return dim


def _convert_dims(dimensions: List[str], direction: str) -> List[str]:
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

def _has_axis(rai: 'RandomAccessibleInterval'):
    """Check if a RandomAccessibleInterval has axes.
    """
    return hasattr(rai, 'axis')


def _is_arraylike(arr):
    """Check if object is an array.
    """
    return hasattr(arr, 'shape') and \
        hasattr(arr, 'dtype') and \
        hasattr(arr, '__array__') and \
        hasattr(arr, 'ndim')


def _is_xarraylike(xarr):
    """Check if object is an xarray.
    """
    return hasattr(xarr, 'values') and \
        hasattr(xarr, 'dims') and \
        hasattr(xarr, 'coords') and \
        _is_arraylike(xarr.values)


def _to_pydim(key: str) -> str:
    pydims = {
        "Time" : "t",
        "slice" : "pln",
        "Z" : "pln",
        "Y" : "row",
        "X" : "col",
        "Channel" : "ch",
    }

    if key in pydims:
        return pydims[key]
    else:
        return key


def _to_ijdim(key: str) -> str:
    ijdims = {
        "col" : "X",
        "x" : "X",
        "row" : "Y",
        "y" : "Y",
        "ch" : "Channel",
        "c" : "Channel",
        "pln" : "Z",
        "z" : "Z",
        "t": "Time",
    }

    if key in ijdims:
        return ijdims[key]
    else:
        return key