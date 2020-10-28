from __future__ import division

import scyjava_config
import jpype
import jpype.imports
import logging
import jgo
import debugtools as dt
import os
import xarray as xr
import numpy as np

from jpype import JClass, JImplements, JLong, JArray

_logger = logging.getLogger(__name__)

def config_min_ij():
    
    # retrieve endpoints and repos
    endpoints = scyjava_config.get_endpoints()
    repositories = scyjava_config.get_repositories()

    # set endpoints
    scyjava_config.add_endpoints('net.imagej:imagej')
    scyjava_config.add_endpoints('net.imglib2:imglib2-imglyb')
    scyjava_config.add_endpoints('net.imagej:imagej-legacy')

    # set classpath
    if len(endpoints) > 0:
        endpoints = endpoints[:1] + sorted(endpoints[1:])
        _logger.debug('Using endpoints %s', endpoints)
        _, workspace = jgo.resolve_dependencies(
            '+'.join(endpoints),
            m2_repo=scyjava_config.get_m2_repo(),
            cache_dir=scyjava_config.get_cache_dir(),
            manage_dependencies=scyjava_config.get_manage_deps(),
            repositories=repositories,
            verbose=scyjava_config.get_verbose()
        )
        jpype.addClassPath(os.path.join(workspace, '*'))
    
    return

def get_xarr():

    xarr = xr.DataArray(np.random.rand(5, 4, 6, 12, 3), dims=['t', 'z', 'y', 'x', 'c'],
                        coords={'x': list(range(0, 12)), 'y': list(np.arange(0, 12, 2)), 'c': [0, 1, 2],
                                'z': list(np.arange(10, 50, 10)), 't': list(np.arange(0, 0.05, 0.01))},
                        attrs={'Hello': 'Wrld'})

    return xarr

def debug_to_imglib(source):

    address = source.ctypes.data
    stride = np.array(source.stride[::-1]) / source.itemsize

    NumpyToImgLibConversionsWithStride.toDouble(address, tuple(stride), source.shape[::-1])

    return

# config and start JVM
config_min_ij()
jpype.startJVM()

### imglyb code snippet -- begin
NumpyToImgLibConversionsWithStride = JClass('net.imglib2.python.NumpyToImgLibConversionsWithStride')
ReferenceGuardingRandomAccessibleInterval = JClass('net.imglib2.python.ReferenceGuardingRandomAccessibleInterval')

@JImplements('net.imglib2.python.ReferenceGuardingRandomAccessibleInterval$ReferenceHolder')
class ReferenceGuard():

    def __init__(self, *args, **kwargs):
        self.args = args

### imglyb code snippet -- end

# setup ij
ImageJ = JClass('net.imagej.ImageJ')
ij = ImageJ()
dt.print_ij_version(ij)

# debug here
xarr = get_xarr() # ok
print("[DEBUG] xarr type: {0}".format(type(xarr)))
vals = np.moveaxis(xarr.values, -1, 0) # ok
print("[DEBUG] vals type: {0}".format(type(vals)))
print("[DEBUG] vals.dtype: {0}".format(vals.dtype))
ref_guard = ReferenceGuard(vals) # ok
print("[DEBUG] ReferenceGuard: {0}".format(ref_guard))
print("[DEBUG] ReferenceGuard type: {0}".format(type(ref_guard)))
address = vals.ctypes.data # ok
long_address = JLong(address)
print("[DEBUG] address: {0}".format(address))
print("[DEBUG] address type: {0}".format(type(address)))
print("[DEBUG] long_address: {0}".format(long_address))
print("[DEBUG] long_address type: {0}".format(type(long_address)))
stride = np.array(vals.strides[::-1]) / vals.itemsize # ok
print("[DEBUG] stride: {0}".format(stride))
print("[DEBUG] stride type: {0}".format(type(stride)))
long_arr_stride = JArray(JLong)(stride)
print("[DEBUG] long_arr_stride: {0}".format(long_arr_stride))
print("[DEBUG] long_arr_stride type: {0}".format(type(long_arr_stride)))
vals_shape = vals.shape[::-1]
long_arr_vals = JArray(JLong)(vals_shape)
print("[DEBUG] long_arr_vals: {0}".format(long_arr_vals))
print("[DEBUG] long_arr_vals type: {0}".format(type(long_arr_vals)))
x = NumpyToImgLibConversionsWithStride.toDouble(long_address, long_arr_stride, long_arr_vals)
print("[DEBUG] x: {0}".format(x))
