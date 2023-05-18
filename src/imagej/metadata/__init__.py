import scyjava as sj

from imagej._java import jc


def _imgplus_metadata_to_python_metadata(img_metadata, py_metadata: dict = None):
    if py_metadata is None:
        py_metadata = {}

    for k, v in img_metadata.items():
        py_metadata[sj.to_python(k)] = sj.to_python(v)

    return py_metadata


def _python_metadata_to_imgplus_metadata(py_metadata: dict):
    return sj.to_java(py_metadata["imagej"])


def is_rgb_merged(img: "jc.ImgPlus") -> bool:
    """
    Check if the ImgPlus is RGB merged.

    :param img: An input net.imagej.ImgPlus
    :return: bool
    """
    e = img.firstElement()
    # check if signed
    if e.getMinValue() < 0:
        return False
    # check if integer type
    if not isinstance(e, jc.IntegerType):
        return False
    # check if bits per pixel is 8
    if e.getBitsPerPixel() != 8:
        return False
    # check for channel dimension (returns -1 if missing)
    ch_index = img.dimensionIndex(jc.Axes.CHANNEL)
    if ch_index < 0:
        return False
    # check if channel dimension is size 3 (RGB)
    if img.dimension(ch_index) != 3:
        return False

    return True


def create_xarray_metadata(img: "jc.ImgPlus") -> dict:
    """
    Create the ImageJ xarray.DataArray metadata.

    :param img: Input net.imagej.ImgPlus.
    :retutn: A Python dict representing the ImageJ metadata.
    """
    # create empty dict for metadata
    py_metadata = {}

    # try to get ImgPlus metadata
    try:
        img_metadata = img.getProperties()
    except AttributeError:
        img_metadata = None

    # convert metadata to python and add to dict
    if img_metadata is not None:
        py_metadata = _imgplus_metadata_to_python_metadata(img_metadata, py_metadata)

    # add additional metadata
    py_metadata["RGB"] = is_rgb_merged(img)

    return py_metadata
