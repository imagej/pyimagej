# Python wrapper for ImageJ

[`pyimagej`](https://github.com/imagej/pyimagej) provides a set of
wrapper functions for integration between ImageJ and Python.

It also provides a high-level entry point `imagej.IJ` for invoking
[ImageJ Server](https://github.com/imagej/imagej-server) APIs;
see "ImageJ Server" below for details.

## Installation

1. Install [Conda](https://conda.io/):
    * On Windows, install Conda using [Chocolatey](https://chocolatey.org): `choco install miniconda3`
    * On macOS, install Conda using [Homebrew](https://brew.sh): `brew install miniconda`
    * On Linux, install Conda using its [RPM or Debian package](https://www.anaconda.com/rpm-and-debian-repositories-for-miniconda/), or [with the Miniconda install script](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html).

2. [Activate the conda-forge channel](https://conda-forge.org/docs/user/introduction.html#how-can-i-install-packages-from-conda-forge):
    ```
    conda config --add channels conda-forge
    conda config --set channel_priority strict
    ```

3. Install pyimagej into a new conda environment:
    ```
    conda create -n pyimagej pyimagej openjdk=8
    ```

4. Whenever you want to use pyimagej, activate its environment:
    ```
    conda activate pyimagej
    ```

### Installation asides

* If you want to use [scikit-image](https://scikit-image.org/) in conjunction,
  as demonstrated below, you can install it also via:
    ```
    conda install scikit-image
    ```

* The above command installs pyimagej with OpenJDK 8; if you leave off the
  `openjdk=8` it will install OpenJDK 11 by default, which should also work, but
  is less well tested and may have more rough edges.

* It is possible to dynamically install pyimagej from within a Jupyter notebook:
    ```
    import sys
    !conda install --yes --prefix {sys.prefix} -c conda-forge pyimagej openjdk=8
    ```
  This approach is useful for [JupyterHub](https://jupyter.org/hub) on the
  cloud, e.g. [Binder](https://mybinder.org/), to utilize pyimagej in select
  notebooks without advance installation. This reduces time needed to create
  and launch the environment, at the expense of a longer startup time the first
  time a pyimagej-enabled notebook is run. See [this itkwidgets example
  notebook](https://github.com/InsightSoftwareConsortium/itkwidgets/blob/v0.24.2/examples/ImageJImgLib2.ipynb)
  for an example.

* It is possible to dynamically install pyimagej on
  [Google Colab](https://colab.research.google.com/). See
  [this thread](https://forum.image.sc/t/pyimagej-on-google-colab/32804) for
  guidance. A major advantage of Google Colab is free GPU in the cloud.

* If you would prefer to install pyimagej via pip, more legwork is required.
  See [this thread](https://forum.image.sc/t/how-do-i-install-pyimagej/23189/4)
  for hints.

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

#### With more memory available to Java

Java's virtual machine (the JVM) has a "max heap" value limiting how much
memory it can use. You can increase the value as follows:

```python
import scyjava_config
scyjava_config.add_options('-Xmx6g')
import imagej
ij = imagej.init()
```

Replace `6g` with the amount of memory Java should have. You can also pass
[other JVM arguments](https://docs.oracle.com/javase/8/docs/technotes/tools/unix/java.html).

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
