# PyImageJ: Python wrapper for ImageJ2

[![Image.sc Forum](https://img.shields.io/badge/dynamic/json.svg?label=forum&url=https%3A%2F%2Fforum.image.sc%2Ftags%2Fpyimagej.json&query=%24.topic_list.tags.0.topic_count&colorB=brightgreen&suffix=%20topics&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA4AAAAOCAYAAAAfSC3RAAABPklEQVR42m3SyyqFURTA8Y2BER0TDyExZ+aSPIKUlPIITFzKeQWXwhBlQrmFgUzMMFLKZeguBu5y+//17dP3nc5vuPdee6299gohUYYaDGOyyACq4JmQVoFujOMR77hNfOAGM+hBOQqB9TjHD36xhAa04RCuuXeKOvwHVWIKL9jCK2bRiV284QgL8MwEjAneeo9VNOEaBhzALGtoRy02cIcWhE34jj5YxgW+E5Z4iTPkMYpPLCNY3hdOYEfNbKYdmNngZ1jyEzw7h7AIb3fRTQ95OAZ6yQpGYHMMtOTgouktYwxuXsHgWLLl+4x++Kx1FJrjLTagA77bTPvYgw1rRqY56e+w7GNYsqX6JfPwi7aR+Y5SA+BXtKIRfkfJAYgj14tpOF6+I46c4/cAM3UhM3JxyKsxiOIhH0IO6SH/A1Kb1WBeUjbkAAAAAElFTkSuQmCC)](https://forum.image.sc/tag/pyimagej)
[![Build Status](https://github.com/imagej/pyimagej/actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/imagej/pyimagej/actions/workflows/build.yml)
[![codecov](https://codecov.io/gh/imagej/pyimagej/branch/main/graph/badge.svg?token=9z6AYgHINK)](https://codecov.io/gh/imagej/pyimagej)
[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/imagej/pyimagej/main)

PyImageJ provides a set of wrapper functions for integration between [ImageJ2]
and Python. It also supports the original [ImageJ] API and data structures.

A major advantage of this approach is the ability to combine ImageJ and ImageJ2
with other tools available from the Python software ecosystem, including NumPy,
SciPy, scikit-image, [CellProfiler], [OpenCV], [ITK] and many more.

## Quick Start

Jump into the [documentation and tutorials](https://pyimagej.readthedocs.io/) to get started!

## System Requirements

### Hardware Requirements

PyImageJ requires at minimum a standard computer with enough RAM and CPU
performance to support the workflow operations defined by the user. While
PyImageJ will run on a range of hardware, we recommend the following RAM
and CPU specifications:

- RAM: >= 2 GB (64 MB minimum)
- CPU: >= 1 core

Notably, PyImageJ can be installed and used on server infrastructure for
large scale image processing.

### OS Requirements

PyImageJ has been tested on the following operating systems:

- Linux (Ubuntu 20.04 LTS)
- Windows
- macOS

### Software Requirements

PyImageJ requires the following packages:

* Python >= 3.7
* [imglyb] >= 2.0.1
* [jgo] >= 1.0.3
* [JPype] >= 1.3.0
* [labeling] >= 0.1.12
* [matplotlib] \(optional, for `ij.py.show` function only)
* [NumPy]
* [scyjava] >= 1.8.0
* [xarray]

PyImageJ will not function properly if dependency versions are too old.

In addition, PyImageJ requires [OpenJDK] and [Maven] to be installed.

## Installation

PyImageJ can be installed using [Conda]+[Mamba]. Here is how to create
and activate a new conda environment with PyImageJ available:

```
conda install mamba -n base -c conda-forge
mamba create -n pyimagej -c conda-forge pyimagej openjdk=8
conda activate pyimagej
```

Alternately, you can install PyImageJ with pip, but in this
case you will need to install OpenJDK and Maven manually.

Installation time takes approximately 20 seconds. Initializing PyImageJ
takes an additional ~30 seconds to ~2-3 minutes (depending on bandwidth)
while it downloads and caches the needed Java libraries.

For detailed installation instructions and requirements, see
[Installation](https://pyimagej.readthedocs.io/en/latest/Install.html).

## Usage

The first step when using PyImageJ is to create an ImageJ2 gateway.
This gateway can point to any official release of ImageJ2 or to a local
installation. Using the gateway, you have full access to the ImageJ2 API,
plus utility functions for translating between Python (NumPy, xarray,
pandas, etc.) and Java (ImageJ2, ImgLib2, etc.) structures.

For instructions on how to start up the gateway for various settings, see
[How to initialize PyImageJ](https://pyimagej.readthedocs.io/en/latest/Initialization.html).

Here is an example of opening an image using ImageJ2 and displaying it:

```python
# Create an ImageJ2 gateway with the newest available version of ImageJ2.
import imagej
ij = imagej.init()

# Load an image.
image_url = 'https://imagej.net/images/clown.jpg'
jimage = ij.io().open(image_url)

# Convert the image from ImageJ2 to xarray, a package that adds
# labeled datasets to numpy (http://xarray.pydata.org/en/stable/).
image = ij.py.from_java(jimage)

# Display the image (backed by matplotlib).
ij.py.show(image, cmap='gray')
```

For more, see the
[tutorial notebooks](https://pyimagej.readthedocs.io/en/latest/notebooks.html).

## API Reference

For a complete reference of the PyImageJ API, please see the
[API Reference](https://pyimagej.readthedocs.io/en/latest/api.html).

## Getting Help

[The Scientific Community Image Forum](https://forum.image.sc/tag/pyimagej)
is the best place to get general help on usage of PyImageJ, ImageJ2, and any
other image processing tasks. Bugs can be reported to the PyImageJ GitHub
[issue tracker](https://github.com/imagej/pyimagej/issues).

## Contributing

All contributions, reports, and ideas are welcome. Contribution is done
via pull requests onto the pyimagej repository.

Most development discussion takes place on the pyimagej
[GitHub repository](https://github.com/imagej/pyimagej).
You can also reach the developers at the
[Image.sc Zulip chat](https://imagesc.zulipchat.com/#narrow/stream/328100-scyjava).

For details on how to develop the PyImageJ codebase, see
[Development.md](https://pyimagej.readthedocs.io/en/latest/Development.html).

------------------------------------------------------------------------------

[ImageJ2]: https://imagej.net/software/imagej2
[ImageJ]: https://imagej.net/software/imagej
[CellProfiler]: https://imagej.net/software/cellprofiler
[OpenCV]: https://imagej.net/software/opencv
[ITK]: https://imagej.net/software/itk
[imglyb]: https://github.com/imglib/imglyb
[jgo]: https://github.com/scijava/jgo
[JPype]: https://jpype.readthedocs.io/
[labeling]: https://github.com/Labelings/Labeling
[matplotlib]: https://matplotlib.org/
[NumPy]: https://numpy.org/
[scyjava]: https://github.com/scijava/scyjava
[xarray]: https://docs.xarray.dev/
[OpenJDK]: https://en.wikipedia.org/wiki/OpenJDK
[Maven]: https://maven.apache.org/
[Conda]: https://conda.io/
[Mamba]: https://mamba.readthedocs.io/
