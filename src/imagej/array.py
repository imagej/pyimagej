import numpy as np
import xarray as xr
from scyjava import _convert

import imagej.dims as dims


@xr.register_dataarray_accessor("img")
class ImgAccessor:
    def __init__(self, xarr):
        self._data = xarr

    @property
    def is_rgb(self):
        """
        Returns True or False if the xarray.DataArray is an RGB image.

        :return: Boolean
        """
        ch_labels = ["c", "ch", "Channel"]
        # check if array is signed
        if self._data.min() < 0:
            return False
        # check if array is integer dtype
        if not np.issubdtype(self._data.data.dtype, np.integer):
            return False
        # check bitsperpixel
        if self._data.dtype.itemsize * 8 != 8:
            return False
        # check if "channel" present
        if not any(dim in self._data.dims for dim in ch_labels):
            return False
        # check channel length = 3 exactly
        for dim in self._data.dims:
            if dim in ch_labels:
                loc = self._data.dims.index(dim)
                if self._data.shape[loc] != 3:
                    return False

        return True


@xr.register_dataarray_accessor("metadata")
class MetadataAccessor:
    def __init__(self, xarr):
        self._data = xarr
        self._update()

    @property
    def axes(self):
        """
        Returns a tuple of the ImageJ axes.

        :return: A Python tuple of the ImageJ axes.
        """
        return (
            tuple(self._data.attrs["imagej"].get("scifio.metadata.image").get("axes"))
            if "scifio.metadata.image" in self._data.attrs["imagej"]
            else None
        )

    def set(self, metadata: dict):
        """
        Set the metadata of the parent xarray.DataArray.

        :param metadata: A Python dict representing the image metadata.
        """
        self._data.attrs["imagej"] = metadata

    def get(self):
        """
        Get the metadata dict of the the parent xarray.DataArray.

        :return: A Python dict representing the image metadata.
        """
        return self._data.attrs["imagej"]

    def tree(self):
        """
        Print a tree of the metadata of the parent xarray.DataArray.
        """
        self._print_dict_tree(self._data.attrs["imagej"])

    def _print_dict_tree(self, dictionary, indent="", prefix=""):
        for idx, (key, value) in enumerate(dictionary.items()):
            if idx == len(dictionary) - 1:
                connector = "└──"
            else:
                connector = "├──"
            print(indent + connector + prefix + " " + str(key))
            if isinstance(value, (dict, _convert.JavaMap)):
                if idx == len(dictionary) - 1:
                    self._print_dict_tree(value, indent + "    ", prefix="── ")
                else:
                    self._print_dict_tree(value, indent + "│   ", prefix="── ")

    def _update(self):
        if self._data.attrs.get("imagej"):
            # update axes
            axes = [None] * len(self._data.dims)
            for i in range(len(self.axes)):
                ax_label = dims._convert_dim(self.axes[i].type().getLabel(), "python")
                if ax_label in self._data.dims:
                    axes[self._data.dims.index(ax_label)] = self.axes[i]
            self._data.attrs["imagej"].get("scifio.metadata.image", {})["axes"] = axes

            # update axis lengths
            old_ax_len_metadata = (
                self._data.attrs["imagej"]
                .get("scifio.metadata.image", {})
                .get("axisLengths", {})
            )
            new_ax_len_metadata = {}
            for i in range(len(self.axes)):
                ax_type = self.axes[i].type()
                if ax_type in old_ax_len_metadata.keys():
                    ax_label = dims._convert_dim(ax_type.getLabel(), "python")
                    curr_ax_len = self._data.shape[self._data.dims.index(ax_label)]
                    new_ax_len_metadata[ax_type] = curr_ax_len
            self._data.attrs["imagej"].get("scifio.metadata.image", {})[
                "axisLengths"
            ] = new_ax_len_metadata
