# PyImageJ: Python wrapper for ImageJ
[![Build Status](https://travis-ci.org/imagej/pyimagej.svg?branch=master)](https://travis-ci.org/imagej/pyimagej)

## ! WARNING: This branch is testing [JPype](https://jpype.readthedocs.io/en/latest/) !

[PyImageJ](https://github.com/imagej/pyimagej) provides a set of
wrapper functions for integration between ImageJ and Python.
A major advantage of this approach is the ability to combine ImageJ with other tools 
available from the Python software ecosystem, including [NumPy](https://www.numpy.org/), 
[SciPy](https://www.scipy.org/), [scikit-image](https://scikit-image.org/), 
[CellProfiler](https://cellprofiler.org/), [OpenCV](https://opencv.org/), 
[ITK](https://itk.org/) and more.

## Installation
PyImageJ can be installed using conda.  Here is how to create and activate a new conda environment holding PyImageJ

```
conda create -n pyimagej -c conda-forge pyimagej openjdk=8
conda activate pyimagej
```

For detailed installation instructions and requirements, see [Install.md](doc/Install.md)

## Usage
The basic usage of PyImageJ is to start up an ImageJ gateway that translates between Python and ImageJ/Java structures.
This gateway can point to any official release of ImageJ or to a local installation.  

For instructions on how to start up the gateway for various settings see 
[Initialization.md](doc/Initialization.md)

Once you have your ImageJ gateway, you can start using it. Here is an example of opening an image using ImageJ and 
displaying it via a convenience function that calls matplotlib:

```python
import imagej
ij = imagej.init()

url_colony = 'https://samples.fiji.sc/new-lenna.jpg' 

# Load the image
lenna = ij.io().open(url_colony)

# Send it to xarray.  Xarray is a package that adds labeled datasets to numpy (http://xarray.pydata.org/en/stable/)
xarray_lenna = ij.py.from_java(lenna)

# Display the image
ij.py.show(xarray_lenna, cmap='gray')
```

For instructions on how to do certain tasks, see [USAGE.md](doc/Usage.md)


# Getting Help 
[The Scientific Community Image Forum](https://forum.image.sc) is the best place to get general help on usage of PyImageJ,
ImageJ, and any other image processing tasks.  Bugs can be reported to the PyImageJ GitHub 
[issue tracker](issues).

# Contributing
All contributions, reports, and ideas are welcome.  Contribution is done via pull requests onto the PyImageJ repository.

Most development discussion takes place on the PyimageJ [GitHub repository](https://github.com/imagej/pyimagej).
You can also reach the developers at the
[imagej gitter](https://gitter.im/imagej/imagej).



