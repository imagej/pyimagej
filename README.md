# PyImageJ: Python wrapper for ImageJ

[![Build Status](https://travis-ci.org/imagej/pyimagej.svg?branch=master)](https://travis-ci.org/imagej/pyimagej)

[PyImageJ](https://github.com/imagej/pyimagej) provides a set of
wrapper functions for integration between ImageJ and Python.
A major advantage of this approach is the ability to combine ImageJ with other tools
available from the Python software ecosystem, including [NumPy](https://www.numpy.org/),
[SciPy](https://www.scipy.org/), [scikit-image](https://scikit-image.org/),
[CellProfiler](https://cellprofiler.org/), [OpenCV](https://opencv.org/),
[ITK](https://itk.org/) and more.

PyImageJ provides a set of wrapper functions for integration between ImageJ
and Python. A major advantage of this approach is the ability to combine
ImageJ with other tools available from the Python software ecosystem,
including NumPy, SciPy, scikit-image, CellProfiler, OpenCV, ITK and more.

## Installation

PyImageJ can be installed using conda. Here is how to create and activate
a new conda environment with PyImageJ available:

```
conda create -n pyimagej -c conda-forge pyimagej openjdk=8
conda activate pyimagej
```

For detailed installation instructions and requirements, see
[Install.md](doc/Install.md).

## Usage

The first step when using PyImageJ is to create an ImageJ gateway.
This gateway can point to any official release of ImageJ or to a local
installation. Using the gateway, you have full access to the ImageJ API,
plus utility functions for translating between Python (NumPy, xarray,
pandas, etc.) and Java (ImageJ, ImgLib2, etc.) structures.

For instructions on how to start up the gateway for various settings see
[Initialization.md](doc/Initialization.md).

Here is an example of opening an image using ImageJ and displaying it:

```python
# Create an ImageJ gateway with the newest available version of ImageJ.
import imagej
ij = imagej.init()

# Load an image.
image_url = 'https://samples.fiji.sc/new-lenna.jpg'
jimage = ij.io().open(image_url)

# Convert the image from ImageJ to xarray, a package that adds
# labeled datasets to numpy (http://xarray.pydata.org/en/stable/).
image = ij.py.from_java(jimage)

# Display the image (backed by matplotlib).
ij.py.show(image, cmap='gray')
```

For instructions on how to do certain tasks, see [USAGE.md](doc/Usage.md)

## Getting Help

[The Scientific Community Image Forum](https://forum.image.sc/tag/pyimagej)
is the best place to get general help on usage of PyImageJ, ImageJ, and any
other image processing tasks. Bugs can be reported to the PyImageJ GitHub
[issue tracker](issues).

## Contributing

All contributions, reports, and ideas are welcome. Contribution is done
via pull requests onto the PyImageJ repository.

Most development discussion takes place on the PyImageJ
[GitHub repository](https://github.com/imagej/pyimagej).
You can also reach the developers at the
[pyimagej gitter](https://gitter.im/imagej/pyimagej).
