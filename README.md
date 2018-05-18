# Python client for imagej

[`imagej.py`](https://github.com/imagej/imagej.py.git) provides a a set of wrapper funtions for integration between imagej and python. It also provides a high-level entry point `imagej.IJ` for invoking `imagej-server` APIs.

## Requirements:

    default:
    - pyjnius
    - imglib2-imglyb

Refer to Pyjnius installation guide for installing [Pyjnius](http://pyjnius.readthedocs.io/en/latest/installation.html).

For imglib2-imglyb installation, you can simply use `conda install -c hanslovsky imglib2-imglyb` Other infomation regarding imglyb can be found in imglyb [git repo](https://github.com/hanslovsky/imglib2-imglyb).

    imagej_server:
    - requests
    - Pillow

Use `pip install -r server_requirements.txt` to install requirements for server.

`Pillow` is required for the `IJ.show()` function. In addition, `display` or `xv` needs to exist in your system to view the image.

## Usage

```python
# Spin up the ImageJ context.
import imagej
imagej.quiet_init('/Applications/Fiji.app')
import imglyb
from jnius import autoclass
ImageJ = autoclass('net.imagej.ImageJ')
ij = ImageJ()

# Import an image with scikit-image.
import skimage
from skimage import io
# NB: Blood vessel image from: https://www.fi.edu/heart/blood-vessels
img = io.imread('https://www.fi.edu/sites/default/files/General_EduRes_Heart_BloodVessels_0.jpg')
import numpy as np
img = np.mean(img, axis=2)

# Invoke ImageJ's Frangi vesselness op.
vessels = np.zeros(img.shape, dtype=img.dtype)
ij.op().filter().frangiVesselness(imglyb.to_imglib(vessels), imglyb.to_imglib(img), [1, 1], 20)
```

For imagej-server, there is a short usage example [here](https://github.com/kkangle/imagej.py/tree/pyjinus/imagej/server).
