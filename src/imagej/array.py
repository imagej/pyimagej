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
        return
