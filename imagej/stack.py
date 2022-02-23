from jpype import JArray, JLong
import scyjava as sj

def rai_slice(rai, imin: tuple, imax: tuple):
    """Slice ImgLib2 images.

    Slice ImgLib2 images using Python's slice notation to define the
    desired slice range. Returned interval includes both imin and imax

    :param rai: An ImgLib2 RandomAccessibleInterval
    :param imin: Tuple of minimum interval range values.
    :param imax: Tuple of maximum interval range values.
    :return: Sliced ImgLib2 RandomAccisbleInterval.
    """
    Views = sj.jimport('net.imglib2.view.Views')
    dims = _get_dims(rai)
    imin_fix = JArray(JLong)(len(dims))
    imax_fix = JArray(JLong)(len(dims))

    dim_itr = range(len(dims))
    for py_dim, j_dim in zip(dim_itr, reversed(dim_itr)):

        # Set minimum
        if imin[py_dim] == None:
            index = 0
        else:
            index = imin[py_dim]
        imin_fix[j_dim] = JLong(index % dims[j_dim])
        # Set maximum
        if imax[py_dim] == None:
            index = (dims[j_dim] - 1)
        else:
            index = imax[py_dim]
        imax_fix[j_dim] = JLong(index % dims[j_dim])

    return Views.dropSingletonDimensions(Views.interval(rai, imin_fix, imax_fix))

def _get_dims(image):
    """Get ImgLib2 image dimensions."""
    if isinstance(image, sj.jimport('net.imglib2.RandomAccessibleInterval')):
        return tuple(sj.jimport('net.imglib2.util.Intervals').dimensionsAsLongArray(image))