import xarray as xr


@xr.register_dataarray_accessor("img")
class ImgAccessor:
    def __init__(self, xarr):
        self._data = xarr

    @property
    def is_rgb(self):
        return


@xr.register_dataarray_accessor("metadata")
class MetadataAccessor:
    def __init__(self, xarr):
        self._data = xarr
        self._metadata = None

    @property
    def axes(self):
        """
        Returns a tuple of the ImageJ axes.

        :return: A Python tuple of the ImageJ axes.
        """
        return (
            tuple(self._metadata.get("scifio.metadata.image").get("axes"))
            if "scifio.metadata.image" in self._metadata
            else None
        )

    def set(self, metadata: dict):
        """
        Set the metadata of the parent xarray.DataArray.

        :param metadata: A Python dict representing the image metadata.
        """
        self._metadata = metadata

    def get(self):
        """
        Get the metadata dict of the the parent xarray.DataArray.

        :return: A Python dict representing the image metadata.
        """
        return self._metadata
