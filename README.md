# Python wrapper for ImageJ

[`pyimagej`](https://github.com/imagej/pyimagej) provides a set of
wrapper functions for integration between ImageJ and Python.

It also provides a high-level entry point `imagej.IJ` for invoking
[ImageJ Server](https://github.com/imagej/imagej-server) APIs;
see "ImageJ Server" below for details.

## Installation

The recommended way to install `pyimagej` is with [Conda](https://conda.io/):

```
conda config --add channels conda-forge 
conda install pyimagej openjdk=8
```

The above installs it with OpenJDK 8; if you leave off the `openjdk=8` it will install OpenJDK 11 by default, which should also work, but is less well tested and may have more rough edges.

## Usage

### Quick start

See [this Jupyter notebook](https://nbviewer.jupyter.org/github/imagej/tutorials/blob/master/notebooks/1-Using-ImageJ/6-ImageJ-with-Python-Kernel.ipynb).

### Creating the ImageJ gateway

#### Newest available version

If you want to launch the newest available release version of ImageJ:

```python
import imagej
ij = imagej.init()
```

This invocation will automatically download and cache the newest release of
[net.imagej:imagej](http://maven.imagej.net/#nexus-search;gav~net.imagej~imagej~~~).

#### Explicitly specified version

You can specify a particular version, to facilitate reproducibility:

```python
import imagej
ij = imagej.init('2.0.0-rc-68')
ij.getVersion()
```

#### With graphical capabilities

If you want to have support for the graphical user interface:

```python
import imagej
ij = imagej.init(headless=False)
ij.ui().showUI()
```

Note there are issues with Java AWT via Python on macOS; see
[this article](https://github.com/imglib/imglyb#awt-through-pyjnius-on-osx)
for a workaround.

#### Including ImageJ 1.x support

By default, the ImageJ gateway will not include the
[legacy layer](https://imagej.net/Legacy) for backwards compatibility with
[ImageJ 1.x](https://imagej.net/ImageJ1).
You can enable the legacy layer as follows:

```python
import imagej
ij = imagej.init('net.imagej:imagej+net.imagej:imagej-legacy')
```

#### Including Fiji plugins

By default, the ImageJ gateway will include base ImageJ2 functionality only,
without additional plugins such as those that ship with the
[Fiji](https://fiji.sc/) distribution of ImageJ.

You can create an ImageJ gateway including Fiji plugins as follows:

```python
import imagej
ij = imagej.init('sc.fiji:fiji')
```

#### From a local installation

If you have an installation of [ImageJ2](https://imagej.net/ImageJ2)
such as [Fiji](https://fiji.sc/), you can wrap an ImageJ gateway around it:

```python
import imagej
ij = imagej.init('/Applications/Fiji.app')
```

Replace `/Applications/Fiji.app` with the actual location of your installation.

### Using the ImageJ gateway

Once you have your ImageJ gateway, you can start using it. Here is an example:

```python
# Import an image with scikit-image.
import skimage
from skimage import io
# NB: Blood vessel image from: https://www.fi.edu/heart/blood-vessels
img = io.imread('https://www.fi.edu/sites/fi.live.franklinds.webair.com/files/styles/featured_large/public/General_EduRes_Heart_BloodVessels_0.jpg')
import numpy as np
img = np.mean(img, axis=2)

# Invoke ImageJ's Frangi vesselness op.
vessels = np.zeros(img.shape, dtype=img.dtype)
ij.op().filter().frangiVesselness(ij.py.to_java(vessels), ij.py.to_java(img), [1, 1], 20)
```

See also `test/test_imagej.py` for other examples of usage.


# ImageJ Server

## Requirements

The imagej.server module has its own requirements:

* `requests` is required to communicate with the ImageJ server.
* `pillow` is required for the `imagej.server.IJ.show()` function.
  In addition, `display` or `xv` must be available to view the image.

## Usage

There is a short usage example
[here](https://github.com/imagej/imagej.py/blob/master/imagej/server/usage.py).
