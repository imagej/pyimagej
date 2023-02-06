"""
Utility functions for manipulating image stacks.
"""
from typing import List, Tuple

import scyjava as sj


def rai_slice(rai, imin: Tuple, imax: Tuple, istep: Tuple):
    """Slice ImgLib2 images.

    Slice ImgLib2 images using Python's slice notation to define the
    desired slice range. Returned interval includes both imin and imax

    :param rai: An ImgLib2 RandomAccessibleInterval
    :param imin: Tuple of minimum interval range values.
    :param imax: Tuple of maximum interval range values.
    :return: Sliced ImgLib2 RandomAccessibleInterval.
    """

    Views = sj.jimport("net.imglib2.view.Views")
    shape = rai.shape
    imin_fix = sj.jarray("j", [len(shape)])
    imax_fix = sj.jarray("j", [len(shape)])
    dim_itr = range(len(shape))

    for py_dim, j_dim in zip(dim_itr, dim_itr):
        # Set minimum
        if imin[py_dim] is None:
            index = 0
        else:
            index = imin[py_dim]
            if index < 0:
                index += shape[j_dim]
        imin_fix[j_dim] = index
        # Set maximum
        if imax[py_dim] is None:
            index = shape[j_dim] - 1
        else:
            index = imax[py_dim]
            if index < 0:
                index += shape[j_dim]
        imax_fix[j_dim] = index

    istep_fix = sj.jarray("j", [istep])

    if _index_within_range(imin_fix, shape) and _index_within_range(imax_fix, shape):
        intervaled = Views.interval(rai, imin_fix, imax_fix)
        stepped = Views.subsample(intervaled, istep_fix)

    # TODO: better match NumPy squeeze behavior. See imagej/pyimagej#1231
    dimension_reduced = Views.dropSingletonDimensions(stepped)
    return dimension_reduced


def _index_within_range(query: List[int], source: List[int]) -> bool:
    """Check if query is within range of source index.
    :param query: List of query int
    :param source: List of soure int
    """
    dim_num = len(query)
    for i in range(dim_num):
        if query[i] > source[i]:
            raise IndexError(
                f"index {query[i]} is out of bound for axis {i} with size {source[i]}"
            )

    return True
