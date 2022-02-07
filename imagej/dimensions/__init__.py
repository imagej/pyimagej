import scyjava as sj
from jpype import JObject

def get_axes_labels(image) -> list:
    """
    Get axis labels
    """
    jaxes = [(JObject(image.axis(idx), sj.jimport('net.imagej.axis.CalibratedAxis')))
                    for idx in range(image.numDimensions())]

    return [_to_lower_dims(jaxes[idx].type().getLabel()) for idx in range(len(jaxes))]


def reorganize(image, dimensions: list):
    """
    Reorignize images order via permute.
    """
    img = _convert_to_imgplus(image)

    # check for dimension count mismatch
    dim_num = image.numDimensions()
    
    if len(dimensions) != dim_num:
        raise ValueError(f"Mismatched dimension coun: {len(dimensions)} != {dim_num}")

    # get ImageJ resources
    ImgPlus = sj.jimport('net.imagej.ImgPlus')
    ImgView = sj.jimport('net.imglib2.img.ImgView')
    Views = sj.jimport('net.imglib2.view.Views')

    # copy dimensional axes into
    axes = []
    for i in range(dim_num):
        old_dim = dimensions[i]
        axes.append(img.axis(old_dim))

    # repeatedly permute the image dimensions into shape
    rai = img.getImg()
    for i in range(dim_num):
        old_dim = dimensions[i]
        if old_dim == i:
            continue
        rai = Views.permute(rai, old_dim, i)

        # update index mapping acccordingly...this is hairy ;-)
        for j in range(dim_num):
            if dimensions[j] == i:
                dimensions[j] = old_dim
                break

        dimensions[i] = i

    return ImgPlus(ImgView.wrap(rai), img.getName(), axes)


def _convert_to_imgplus(image):
    """
    Check if image is Dataset and convert to ImgPlus.
    """
    if isinstance(image, sj.jimport('net.imagej.Dataset')):
        return image.getImgPlus()
    else:
        return image
        

def to_python(dimensions: list, label_output=True) -> list:
    """
    Convert any dimension order to python/numpy order.
    :param dimensions: Lower case single character dimensions.
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


def to_java(dimensions: list) -> list:
    """
    Convert any dimension order to java/imglib2 order.
    :param dimensions: Lower case single character dimenions.
    """
    new_dim_order = []
    java_ref_order = ['x', 'y', 'c', 'z', 't']

    for dim in java_ref_order:
        for i in range(len(dimensions)):
            if dim == dimensions[i]:
                new_dim_order.append(dim)

    for i in range(len(dimensions)):
        if dimensions[i] not in java_ref_order:
            new_dim_order.insert(1, dimensions[i])

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