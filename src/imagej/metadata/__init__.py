from typing import Sequence

import imagej.dims as dims
import imagej.metadata.axis as axis
from imagej._java import jc


def create_imagej_metadata(
    axes: Sequence["jc.CalibratedAxis"], dim_seq: Sequence[str]
) -> dict:
    """
    Create the ImageJ metadata attribute dictionary for xarray's global attributes.
    :param axes: A list or tuple of ImageJ2 axis objects
        (e.g. net.imagej.axis.DefaultLinearAxis).
    :param dim_seq: A list or tuple of the dimension order (e.g. ['X', 'Y', 'C']).
    :return: Dict of image metadata.
    """
    ij_metadata = {}
    if len(axes) != len(dim_seq):
        raise ValueError(
            f"Axes length ({len(axes)}) does not match \
                dimension length ({len(dim_seq)})."
        )

    for i in range(len(axes)):
        # get CalibratedAxis type as string (e.g. "EnumeratedAxis")
        ij_metadata[
            dims._to_ijdim(dim_seq[i]) + "_cal_axis_type"
        ] = axis.calibrated_axis_to_str(axes[i])
        # get scale and origin for DefaultLinearAxis
        if isinstance(axes[i], jc.DefaultLinearAxis):
            ij_metadata[dims._to_ijdim(dim_seq[i]) + "_scale"] = float(axes[i].scale())
            ij_metadata[dims._to_ijdim(dim_seq[i]) + "_origin"] = float(
                axes[i].origin()
            )

    return ij_metadata
