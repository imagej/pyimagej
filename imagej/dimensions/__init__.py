import scyjava as sj
from jpype import JObject
from typing import List, Union

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
    return [_to_lower_dims(axes[idx].type().getLabel()) for idx in range(len(axes))]


def get_dims(image):
    if hasattr(image, 'axis'):
        axes = get_axes(image)
        return get_axes_labels(axes)
    else:
        axes = image.dimensionsAsLongArray()
        return axes


def get_shape(image):
    """
    Get the dimensions of an image
    """
    if _is_arraylike(image):
        return image.shape
    if not sj.isjava(image):
        raise TypeError('Unsupported type: ' + str(type(image)))
    if isinstance(image, sj.jimport('net.imglib2.Dimensions')):
        return [image.dimension(d) for d in range(image.numDimensions())]
    if isinstance(image, sj.jimport('ij.ImagePlus')):
        return image.getDimensions()
    raise TypeError(f'Unsupported Java type: {str(sj.jclass(image).getName())}')


def reorganize(image, permute_order: list):
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


def _convert_to_imgplus(image):
    """
    Check if image is Dataset and convert to ImgPlus.
    """
    if isinstance(image, sj.jimport('net.imagej.Dataset')):
        return image.getImgPlus()
    else:
        return image


def to_python_order(dimensions: List[str], label_output=True) -> List[Union[str, int]]:
    """
    Convert any dimension order to python/numpy order.
    :param dimensions: Lower case single character dimensions.
    :param label_output: 
        Boolean that sets whether the resulting list contains the dimension
        labels (e.g. 'x', 'y', ) or numpy style transposition order (e.g. 3, 0, 1).
    """
    new_dim_order = []
    python_ref_order = ['t', 'z', 'y', 'x', 'c']

    for dim in python_ref_order:
        for i in range(len(dimensions)):
            if dim == dimensions[i]:
                if label_output:
                    new_dim_order.append(dim)
                else:
                    new_dim_order.append(i)

    for i in range(len(dimensions)):
        if dimensions[i] not in python_ref_order:
            if label_output:
                new_dim_order.insert(1 ,dimensions[i])
            else:
                new_dim_order.insert(1, i)

    return new_dim_order


def to_java_order(dimensions: list, label_output=True) -> list:
    """
    Convert any dimension order to java/imglib2 order.
    :param dimensions: Lower case single character dimenions.
    :param label_output: 
        Boolean that sets whether the resulting list contains the dimension
        labels (e.g. 'x', 'y', ) or numpy style transposition order (e.g. 3, 0, 1).
    """
    new_dim_order = []
    java_ref_order = ['x', 'y', 'c', 'z', 't']

    for dim in java_ref_order:
        for i in range(len(dimensions)):
            if dim == dimensions[i]:
                if label_output:
                    new_dim_order.append(dim)
                else:
                    new_dim_order.append(i)

    for i in range(len(dimensions)):
        if dimensions[i] not in java_ref_order:
            if label_output:
                new_dim_order.insert(1, dimensions[i])
            else:
                new_dim_order.insert(1, i)

    return new_dim_order


def _to_lower_dims(axis):
    if str(axis) in ['X', 'Y', 'Z', 'C', 'T']:
        return str(axis).lower()
    elif str(axis) =='Channel':
        return 'c'
    elif str(axis) == 'Time':
        return 't'
    else:
        return str(axis)


def _to_upper_dims(axis):
    if str(axis) in ['x', 'y', 'z', 'c', 't']:
        return str(axis).upper()
    else:
        return str(axis)


def _is_arraylike(arr):
    return hasattr(arr, 'shape') and \
        hasattr(arr, 'dtype') and \
        hasattr(arr, '__array__') and \
        hasattr(arr, 'ndim')


def _is_xarraylike(self, xarr):
    return hasattr(xarr, 'values') and \
        hasattr(xarr, 'dims') and \
        hasattr(xarr, 'coords') and \
        self._is_arraylike(xarr.values)