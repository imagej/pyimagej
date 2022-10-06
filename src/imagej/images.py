"""
Utility functions for creating and working with images.
"""


def is_arraylike(arr):
    return (
        hasattr(arr, "shape")
        and hasattr(arr, "dtype")
        and hasattr(arr, "__array__")
        and hasattr(arr, "ndim")
    )


def is_memoryarraylike(arr):
    return (
        is_arraylike(arr)
        and hasattr(arr, "data")
        and type(arr.data).__name__ == "memoryview"
    )


def is_xarraylike(xarr):
    return (
        hasattr(xarr, "values")
        and hasattr(xarr, "dims")
        and hasattr(xarr, "coords")
        and is_arraylike(xarr.values)
    )
