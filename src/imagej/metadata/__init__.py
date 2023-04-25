from typing import Sequence

import imagej.metadata.axis as axis
from imagej._java import jc


def _create_axis_metadata(img: "jc.ImgPlus") -> Sequence[dict]:
    meta_arr = []
    axes = list(img.dim_axes)[::-1]
    dims = list(img.dims)[::-1]
    shape = list(img.shape)[::-1]

    for i in range(len(dims)):
        ax = axes[i]
        ax_meta = {}
        # get per axis metadata
        ax_meta["label"] = dims[i]
        ax_meta["length"] = shape[i]
        ax_meta["CalibratedAxis"] = axis.calibrated_axis_to_str(ax)
        if hasattr(ax, "scale"):
            ax_meta["scale"] = float(ax.scale())
        if hasattr(ax, "origin"):
            ax_meta["origin"] = float(ax.origin())
        # store metadata
        meta_arr.append(ax_meta)

    return meta_arr


def create_imagej_metadata(img: "jc.ImgPlus") -> dict:
    ij_metadata = {}
    # imagej metadata sections
    ij_metadata["axis"] = _create_axis_metadata(img)

    return ij_metadata
