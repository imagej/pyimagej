import scyjava as sj

def rai_slice(rai, imin: tuple, imax: tuple):
    """Slice ImgLib2 images.

    Slice ImgLib2 images using Python's slice notation to define the
    desired slice range.

    :param rai: An ImgLib2 RandomAccessibleInterval
    :param imin: Tuple of minimum interval range values.
    :param imax: Tuple of maximum interval range values.
    :return: Sliced ImgLib2 RandomAccisbleInterval.
    """
    Views = sj.jimport('net.imglib2.view.Views')
    imin_fix = []
    imax_fix = []
    dims = _get_dims(rai)

    for i in range(len(dims)):
        if imin[i] == None:
            imin_fix.append(0)
        if imax[i] == None:
            imax_fix.append(dims[i] - 1)
        else:
            imin_fix.append(imin[i])
            imax_fix.append(imax[i])

    return Views.interval(rai, imin_fix, imax_fix)

def _get_dims(image):
    """Get ImgLib2 image dimensions."""
    if isinstance(image, sj.jimport('net.imglib2.RandomAccessibleInterval')):
        return tuple(sj.jimport('net.imglib2.util.Intervals').dimensionsAsLongArray(image))