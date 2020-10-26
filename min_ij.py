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
import imagej

from jpype import JClass

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

# imglib
NumpyToImgLibConversionsWithStride = JClass('net.imglib2.python.NumpyToImgLibConversionsWithStride')

# setup ij
#ImageJ = JClass('net.imagej.ImageJ')
#ij = ImageJ()
ij = imagej.init()
dt.print_ij_version(ij)

# debug here
xarr = get_xarr()
print("[DEBUG] xarr:")
print(xarr)
print("[DEBUG] xarr type: {0}".format(type(xarr)))
print("[DEBUG] xarr dir:")
print(dir(xarr))

# convert to dataset
ds = ij.py.to_dataset(xarr)
print("[DEBUG] ds:")
print(ds)
print("[DEBUG] ds type: {0}".format(type(ds)))
print("[DEBUG] ds dir:")
print(dir(ds))

