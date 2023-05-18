from typing import Sequence
import scyjava as sj
from imagej._java import jc


def _imgplus_metadata_to_python_metadata(img_metadata, py_metadata: dict = None):
    if py_metadata is None:
        py_metadata = {}

    for k, v in img_metadata.items():
        py_metadata[sj.to_python(k)] = sj.to_python(v)

    return py_metadata


def _python_metadata_to_imgplus_metadata(py_metadata: dict):
    return sj.to_java(py_metadata['imagej'])


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
    py_metadata['RGB'] = is_rgb_merged(img)

    return py_metadata

