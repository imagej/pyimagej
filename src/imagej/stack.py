import scyjava as sj
from typing import List, Tuple


def rai_slice(rai, imin: Tuple, imax: Tuple, istep: Tuple):
    """Slice ImgLib2 images.

    Slice ImgLib2 images using Python's slice notation to define the
    desired slice range. Returned interval includes both imin and imax

    :param rai: An ImgLib2 RandomAccessibleInterval
    :param imin: Tuple of minimum interval range values.
    :param imax: Tuple of maximum interval range values.
    :return: Sliced ImgLib2 RandomAccisbleInterval.
    """

    # HACK: Avoid importing JLong at global scope.
    # Otherwise, building the sphinx docs in doc/rtd fails with:
    #
    #   Warning, treated as error:
    #   autodoc: failed to determine imagej.stack.JLong (<java class 'JLong'>) to be documented, the following exception was raised:
    #   Java Virtual Machine is not running
    #
    # Which can be reproduced in a REPL like this:
    #
    #   >>> from jpype import JLong
    #   >>> help(JLong)
    #
    # So while the import here is unfortunate, it avoids the issue.
    from jpype import JArray, JLong

    Views = sj.jimport("net.imglib2.view.Views")
    shape = rai.shape
    imin_fix = JArray(JLong)(len(shape))
    imax_fix = JArray(JLong)(len(shape))
    dim_itr = range(len(shape))

    for py_dim, j_dim in zip(dim_itr, dim_itr):

        # Set minimum
        if imin[py_dim] == None:
            index = 0
        else:
            index = imin[py_dim]
            if index < 0:
                index += shape[j_dim]
        imin_fix[j_dim] = JLong(index)
        # Set maximum
        if imax[py_dim] == None:
            index = shape[j_dim] - 1
        else:
            index = imax[py_dim]
            if index < 0:
                index += shape[j_dim]
        imax_fix[j_dim] = JLong(index)

    istep_fix = JArray(JLong)(istep)

    if _index_within_range(imin_fix, shape) and _index_within_range(imax_fix, shape):
        intervaled = Views.interval(rai, imin_fix, imax_fix)
        stepped = Views.subsample(intervaled, istep_fix)

    # TODO: better mach NumPy squeeze behavior. See pyimagej/#1231
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
