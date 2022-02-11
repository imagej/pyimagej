import scyjava as sj
from jpype import JObject
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
        rai_axis_types = []
        for i in range(len(rai_dims)):
            rai_axis_types.append(Axes.get(rai_dims[i]))
        return rai_axis_types
    else:
        print("Unsupported action _get_axis_type")
        return


def get_dims(image) -> List[str]:
    """Get the dimensions of an image.

    Get the dimensions (e.g. TZYXC) of an image.

    :param image: An image (e.g. xarray, ImagePlus, Dataset)
    :return: List of dimensions.
    """
    if _is_xarraylike(image):
        return image.dims
    if hasattr(image, 'axis'):
        axes = get_axes(image)
        return _get_axis_labels(axes)
    else:
        return image.dimensionsAsLongArray()


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
    a reference order (e.g. _python_ref_order).

    :param axis_types: List of 'net.imagej.axis.AxisType' from image.
    :param ref_order: List of 'net.imagej.axis.AxisType' from reference order.
    :return: List of int for permuting a image.
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


def _python_ref_order():
    """Get the Python reference order.

    Get a List of 'AxisType' in the Python/scikitimage
    preferred order. Note that this reference order is
    reversed.
    """
    Axes = sj.jimport('net.imagej.axis.Axes')
    return [Axes.CHANNEL, Axes.X, Axes.Y, Axes.Z, Axes.TIME]


def _to_lower_dims(dimensions: List[str]) -> List[str]:
    """Convert a List of dimensions to upper case.
    """
    return [str(dim).lower() for dim in dimensions]


def _to_upper_dims(dimensions: List[str]) -> List[str]:
    """Convert a List of dimensions to lower case.
    """
    return [str(dim).upper() for dim in dimensions]


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
