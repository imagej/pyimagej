"""
PyImageJ provides a set of wrapper functions for integration between ImageJ
and Python. A major advantage of this approach is the ability to combine
ImageJ with other tools available from the Python software ecosystem,
including NumPy, SciPy, scikit-image, CellProfiler, OpenCV, ITK and more.

The first step when using PyImageJ is to create an ImageJ gateway.
This gateway can point to any official release of ImageJ or to a local
installation. Using the gateway, you have full access to the ImageJ API,
plus utility functions for translating between Python (NumPy, xarray,
pandas, etc.) and Java (ImageJ, ImgLib2, etc.) structures.

Here is an example of opening an image using ImageJ and displaying it:

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
"""
from .imagej import *

