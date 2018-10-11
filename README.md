# Python wrapper for ImageJ

[`imagej.py`](https://github.com/imagej/imagej.py) provides a set of
wrapper functions for integration between ImageJ and Python.

It also provides a high-level entry point `imagej.IJ` for invoking
[ImageJ Server](https://github.com/imagej/imagej-server) APIs;
see "ImageJ Server" below for details.

## Requirements

    default:
    - imglyb

Install imglyb using `conda install -c hanslovsky imglyb`.

Further information regarding imglyb can be found in the
[imglyb GitHub repository](https://github.com/imglib/imglyb).

## Usage

In this example, replace `/Applications/Fiji.app` with the location of your Fiji installation.

```python
# Spin up ImageJ.
import imagej
ij = imagej.init('/Applications/Fiji.app')

# Import an image with scikit-image.
import skimage
from skimage import io
# NB: Blood vessel image from: https://www.fi.edu/heart/blood-vessels
img = io.imread('https://www.fi.edu/sites/fi.live.franklinds.webair.com/files/styles/featured_large/public/General_EduRes_Heart_BloodVessels_0.jpg')
import numpy as np
img = np.mean(img, axis=2)

# Invoke ImageJ's Frangi vesselness op.
vessels = np.zeros(img.shape, dtype=img.dtype)
import imglyb
ij.op().filter().frangiVesselness(imglyb.to_imglib(vessels), imglyb.to_imglib(img), [1, 1], 20)
```

See also `test/test_imagej.py` for other examples of usage.


# ImageJ Server

## Requirements

    imagej_server:
    - requests
    - Pillow

Use `pip install -r server_requirements.txt` to install requirements for server.

`Pillow` is required for the imagej.server module's `IJ.show()` function.
In addition, `display` or `xv` needs to exist in your system to view the image.

## Usage

There is a short usage example
[here](https://github.com/imagej/imagej.py/blob/master/imagej/server/usage.py).
