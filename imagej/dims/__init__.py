import scyjava as sj
from jpype import JObject
from typing import List, Tuple

def get_axes(dataset) -> list:
    """
    Get a list of 'net.imagej.axis.CalibratedAxis'.
    :param image: Input Dataset or RandomAccessibleInterval.
    """
    return [(JObject(dataset.axis(idx), sj.jimport('net.imagej.axis.CalibratedAxis'))) for idx in range(dataset.numDimensions())]


def get_axes_labels(axes) -> list:
    """
    Get the axes labels of a list of CalibratedAxis.
    """
    return [str((axes[idx].type().getLabel())) for idx in range(len(axes))]


def get_axis_types(rai) -> List['AxisType']:
    """
    Get a List of 'AxisType' from a RandomAccessibleInterval. Note that Dataset
    and ImgPlus have axis metadata. Other intervals may not have axis metada, such as
    a PlanarImg.
    :param rai: A RandomAccessibleInterval with axis metadata.
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
    # TODO: add check if xarray
    if hasattr(image, 'axis'):
        axes = get_axes(image)
        return get_axes_labels(axes)
    else:
        axes = image.dimensionsAsLongArray()
        return axes


def get_shape(image) -> List[int]:
    """
    Get the dimensions of an image
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


def reorganize(image, permute_order: List[int]) -> 'ImgPlus':
    """
    Reorignize images order via net.imglib2.view.Views permute.
    :param image: A Dataset or ImgPlus.
    :param permute_order: The order in which to permute/transpose the data.
    """
    img = _convert_to_imgplus(image)

    # check for dimension count mismatch
    dim_num = image.numDimensions()
    
    if len(permute_order) != dim_num:
        raise ValueError(f"Mismatched dimension coun: {len(permute_order)} != {dim_num}")

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


def to_python_axes_order(axis_types: List['AxisType']) -> List[int]:
    """
    Convert any dimension order to python/numpy order.
    :param dimensions: Lower case single character dimensions.
    """
    Axes = sj.jimport('net.imagej.axis.Axes')
    new_dim_order = []
    python_ref_order = [Axes.CHANNEL, Axes.X, Axes.Y, Axes.Z, Axes.TIME]

    for axis in python_ref_order:
        for i in range(len(axis_types)):
            if axis == axis_types[i]:
                new_dim_order.append(i)

    for i in range(len(axis_types)):
        if axis_types[i] not in python_ref_order:
                new_dim_order.insert(1, i)

    return new_dim_order


def _convert_to_imgplus(image):
    """
    Check if image is Dataset and convert to ImgPlus.
    """
    if isinstance(image, sj.jimport('net.imagej.Dataset')):
        return image.getImgPlus()
    else:
        return image


def _to_lower_dims(dimensions: List[str]) -> List[str]:
    return [str(dim).lower() for dim in dimensions]


def _to_upper_dims(dimensions: List[str]) -> List[str]:
    return [str(dim).upper() for dim in dimensions]


def _has_axis(rai):
    return hasattr(rai, 'axis')


def _is_arraylike(arr):
    return hasattr(arr, 'shape') and \
        hasattr(arr, 'dtype') and \
        hasattr(arr, '__array__') and \
        hasattr(arr, 'ndim')


def _is_xarraylike(xarr):
    return hasattr(xarr, 'values') and \
        hasattr(xarr, 'dims') and \
        hasattr(xarr, 'coords') and \
        _is_arraylike(xarr.values)
