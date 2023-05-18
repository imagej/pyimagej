from typing import Sequence
from imagej._java import jc






def create_imagej_metadata(img: "jc.ImgPlus") -> dict:
    ij_metadata = {}
    # imagej metadata sections
    ij_metadata["axis"] = _create_axis_metadata(img)

    return ij_metadata
