"""
PyImageJ provides a set of wrapper functions for integration between
ImageJ+ImageJ2 and Python. A major advantage of this approach is the ability to
combine ImageJ+ImageJ2 with other tools available from the Python software
ecosystem, e.g. NumPy, SciPy, scikit-image, CellProfiler, OpenCV, and ITK.

The first step when using PyImageJ is to create an ImageJ2 gateway.
This gateway can point to any official release of ImageJ2 or to a local
installation. Using the gateway, you have full access to the ImageJ2 API,
plus utility functions for translating between Python (NumPy, xarray,
pandas, etc.) and Java (ImageJ, ImageJ2, ImgLib2, etc.) structures.

Here is an example of opening an image using ImageJ2 and displaying it:

.. highlight:: python
.. code-block:: python

    # Create an ImageJ2 gateway with the newest available version of ImageJ2.
    import imagej
    ij = imagej.init()

    # Load an image.
    image_url = 'https://imagej.net/images/clown.png'
    jimage = ij.io().open(image_url)

    # Convert the image from ImageJ2 to xarray, a package that adds
    # labeled datasets to numpy (http://xarray.pydata.org/en/stable/).
    image = ij.py.from_java(jimage)

    # Display the image (backed by matplotlib).
    ij.py.show(image, cmap='gray')
"""

import logging
import os
import re
import sys
import time
import imglyb
import numpy as np
import scyjava as sj
import xarray as xr
import imagej.stack as stack
import imagej.dims as dims
import subprocess
import threading

from enum import Enum
from labeling import Labeling
from pathlib import Path
from typing import List, Tuple
from functools import lru_cache
from jpype import (
    JArray,
    JException,
    JImplements,
    JImplementationFor,
    JObject,
    JOverride,
    setupGuiEnvironment,
)

from .config import __author__, __version__

_logger = logging.getLogger(__name__)
rai_lock = threading.Lock()

# Enable debug logging if DEBUG environment variable is set.
try:
    debug = os.environ["DEBUG"]
    if debug:
        _logger.setLevel(logging.DEBUG)
        dims._logger.setLevel(logging.DEBUG)
except KeyError as e:
    pass


class Mode(Enum):
    """
    An environment mode for the ImageJ2 gateway.
    See the imagej.init function for  more details.
    """

    GUI = "gui"
    HEADLESS = "headless"
    INTERACTIVE = "interactive"

    def __eq__(self, other):
        return super() == other or self.value == other


class ImageJPython:
    """ImageJ/Python convenience methods.

    This class should not be initialized manually. Upon initialization the
    ImageJPython class is attached to the newly initialized ImageJ2 instance in
    the GatewayAddons class via JPype class customization mechanism:

    https://jpype.readthedocs.io/en/latest/userguide.html#class-customizers
    """

    def __init__(self, ij):
        self._ij = ij
        sj.when_jvm_starts(self._add_converters)

    def active_dataset(self) -> "net.imagej.Dataset":
        """Get the active Dataset image.

        Get the active image as a Dataset from the Dataset service.

        :return: The Dataset corresponding to the active image.
        """
        return self._ij.imageDisplay().getActiveDataset()

    def active_image_plus(self, sync=True) -> "ij.ImagePlus":
        """
        ij.py.active_image_plus() is deprecated.
        Use ij.py.active_imageplus() instead.
        """
        _logger.warning(
            "ij.py.active_image_plus() is deprecated. Use ij.py.active_imageplus() instead."
        )
        return self.active_imageplus(sync)

    def active_imageplus(self, sync=True) -> "ij.ImagePlus":
        """Get the active ImagePlus image.

        Get the active image as an ImagePlus, optionally synchronizing from ImageJ to ImageJ2.

        :param sync: Synchronize the current ImagePlus slice if True.
        :return: The ImagePlus corresponding to the active image.
        """
        imp = self._ij.WindowManager.getCurrentImage()
        if imp is None:
            return None
        if sync:
            self.sync_image(imp)
        return imp

    def active_xarray(self, sync=True) -> xr.DataArray:
        """Get the active image as an xarray.

        Get the active image as an xarray.DataArray, synchronizing from ImageJ to ImageJ2.

        :param sync: Synchronize the current ImagePlus slice if True.
        :return: xarray.DataArray array containing the image data.
        """
        # todo: make the behavior use pure ImageJ2 if legacy is not active

        if ij.legacy and ij.legacy.isActive():
            imp = self.active_imageplus(sync=sync)
            return self.from_java(imp)
        else:
            dataset = self.active_dataset()
            return self.from_java(dataset)

    def argstring(self, args, ij1_style=True):
        """
        Assemble an ImageJ (1.x) argument string from arguments in a dict.

        :param args: A dict of arguments in key/value pairs
        :param ij1_style: True to use implicit booleans in original ImageJ style, or False for explicit booleans in ImageJ2 style
        :return: A string version of the arguments
        """
        formatted_args = []
        for key, value in args.items():
            arg = self._format_argument(key, value, ij1_style)
            if arg is not None:
                formatted_args.append(arg)
        return " ".join(formatted_args)

    def dims(self, image):
        """
        ij.py.dims(image) is deprecated. Use image.shape instead.
        """
        _logger.warning("ij.py.dims(image) is deprecated. Use image.shape instead.")
        if self._is_arraylike(image):
            return image.shape
        if not sj.isjava(image):
            raise TypeError("Unsupported type: " + str(type(image)))
        if isinstance(image, sj.jimport("net.imglib2.Dimensions")):
            return list(image.dimensionsAsLongArray())
        if _ImagePlus() and isinstance(image, _ImagePlus()):
            dims = image.getDimensions()
            dims.reverse()
            dims = [dim for dim in dims if dim > 1]
            return dims
        raise TypeError("Unsupported Java type: " + str(sj.jclass(image).getName()))

    def dtype(self, image_or_type):
        """Get the dtype of the input image as a numpy.dtype object.

        Note: for Java-based images, this is different than the image's dtype
        property, because ImgLib2-based images report their dtype as a subclass
        of net.imglib2.type.Type, and ImagePlus images do not yet implement
        the dtype function (see https://github.com/imagej/pyimagej/issues/194).

        :param image_or_type:
            | A NumPy array.
            | OR A NumPy array dtype.
            | OR An ImgLib2 image ('net.imglib2.Interval').
            | OR An ImageJ2 Dataset ('net.imagej.Dataset').
            | OR An ImageJ ImagePlus ('ij.ImagePlus').

        :return: Input image dtype.
        """
        if type(image_or_type) == np.dtype:
            return image_or_type
        if self._is_arraylike(image_or_type):
            return image_or_type.dtype
        if not sj.isjava(image_or_type):
            raise TypeError("Unsupported type: " + str(type(image_or_type)))

        # -- ImgLib2 types --
        if isinstance(image_or_type, sj.jimport("net.imglib2.type.Type")):
            # fmt: off
            ij2_types = {
                #"net.imglib2.type.logic.BitType":                               "bool",
                "net.imglib2.type.numeric.integer.ByteType":                    "int8",
                "net.imglib2.type.numeric.integer.ByteLongAccessType":          "int8",
                "net.imglib2.type.numeric.integer.ShortType":                   "int16",
                "net.imglib2.type.numeric.integer.ShortLongAccessType":         "int16",
                "net.imglib2.type.numeric.integer.IntType":                     "int32",
                "net.imglib2.type.numeric.integer.IntLongAccessType":           "int32",
                "net.imglib2.type.numeric.integer.LongType":                    "int64",
                "net.imglib2.type.numeric.integer.LongLongAccessType":          "int64",
                "net.imglib2.type.numeric.integer.UnsignedByteType":            "uint8",
                "net.imglib2.type.numeric.integer.UnsignedByteLongAccessType":  "uint8",
                "net.imglib2.type.numeric.integer.UnsignedShortType":           "uint16",
                "net.imglib2.type.numeric.integer.UnsignedShortLongAccessType": "uint16",
                "net.imglib2.type.numeric.integer.UnsignedIntType":             "uint32",
                "net.imglib2.type.numeric.integer.UnsignedIntLongAccessType":   "uint32",
                "net.imglib2.type.numeric.integer.UnsignedLongType":            "uint64",
                "net.imglib2.type.numeric.integer.UnsignedLongLongAccessType":  "uint64",
                #"net.imglib2.type.numeric.ARGBType":                            "argb",
                #"net.imglib2.type.numeric.ARGBLongAccessType":                  "argb",
                "net.imglib2.type.numeric.real.FloatType":                      "float32",
                "net.imglib2.type.numeric.real.FloatLongAccessType":            "float32",
                "net.imglib2.type.numeric.real.DoubleType":                     "float64",
                "net.imglib2.type.numeric.real.DoubleLongAccessType":           "float64",
                #"net.imglib2.type.numeric.complex.ComplexFloatType":            "cfloat32",
                #"net.imglib2.type.numeric.complex.ComplexFloatLongAccessType":  "cfloat32",
                #"net.imglib2.type.numeric.complex.ComplexDoubleType":           "cfloat64",
                #"net.imglib2.type.numeric.complex.ComplexDoubleLongAccessType": "cfloat64",
            }
            # fmt: on
            for c in ij2_types:
                if isinstance(image_or_type, sj.jimport(c)):
                    return np.dtype(ij2_types[c])
            raise TypeError(f"Unsupported ImgLib2 type: {image_or_type}")

        # -- ImgLib2 images --
        if isinstance(image_or_type, sj.jimport("net.imglib2.IterableInterval")):
            ij2_type = image_or_type.firstElement()
            return self.dtype(ij2_type)
        if isinstance(image_or_type, _RandomAccessibleInterval()):
            Util = sj.jimport("net.imglib2.util.Util")
            ij2_type = Util.getTypeFromInterval(image_or_type)
            return self.dtype(ij2_type)

        # -- Original ImageJ images --
        if _ImagePlus() and isinstance(image_or_type, _ImagePlus()):
            ij1_type = image_or_type.getType()
            ij1_types = {
                _ImagePlus().GRAY8: "uint8",
                _ImagePlus().GRAY16: "uint16",
                _ImagePlus().GRAY32: "float32",  # NB: ImageJ's 32-bit type is float32, not uint32.
            }
            for t in ij1_types:
                if ij1_type == t:
                    return np.dtype(ij1_types[t])
            raise TypeError(f"Unsupported original ImageJ type: {ij1_type}")

        raise TypeError(
            "Unsupported Java type: " + str(sj.jclass(image_or_type).getName())
        )

    def from_java(self, data):
        """Convert supported Java data into Python equivalents.

        Converts Java objects (e.g. 'net.imagej.Dataset') into the Python
        equivalents.

        :param data: Java object to be converted into its respective Python counterpart.
        :return: A Python object converted from Java.
        """
        # todo: convert a dataset to xarray
        return sj.to_python(data)

    def initialize_numpy_image(self, image) -> np.ndarray:
        """Initialize a NumPy array with zeros and shape of the input image.

        Initialize a new NumPy array with the same dtype and shape as the input
        image with zeros.

        :param image: A RandomAccessibleInterval or NumPy image
        :return:
            A NumPy array with the same dtype and shape as the input
            image, filled with zeros.
        """
        try:
            dtype_to_use = self.dtype(image)
        except TypeError:
            dtype_to_use = np.dtype("float64")

        # get shape of image and invert
        shape = list(image.shape)

        # reverse shape if image is a RandomAccessibleInterval
        if isinstance(image, _RandomAccessibleInterval()):
            shape.reverse()

        return np.zeros(shape, dtype=dtype_to_use)

    def jargs(self, *args):
        """Convert Python arguments into a Java Object[]

        Converts Python arguments into a Java Object[] (i.e.: array of Java
        objects). This is particularly useful in combination with ImageJ2's
        various run functions, including ij.command().run(...),
        ij.module().run(...), ij.script().run(...), and ij.op().run(...).

        :param args: The Python arguments to wrap into an Object[].
        :return: A Java Object[]
        """
        return _JObjectArray()([self.to_java(arg) for arg in args])

    def new_numpy_image(self, image):
        """
        ij.py.new_numpy_image() is deprecated.
        Use ij.py.initialize_numpy_image() instead.
        """
        try:
            dtype_to_use = self.dtype(image)
        except TypeError:
            dtype_to_use = np.dtype("float64")
        _logger.warning(
            "ij.py.new_numpy_image() is deprecated. Use ij.py.initialize_numpy_image() instead."
        )
        return np.zeros(self.dims(image), dtype=dtype_to_use)

    def rai_to_numpy(
        self, rai: "net.imglib2.RandomAccessibleInterval", numpy_array: np.ndarray
    ) -> np.ndarray:
        """Copy a RandomAccessibleInterval into a numpy array.

        The input RandomAccessibleInterval is copied into the pre-initialized numpy array
        with either "fast copy" via 'net.imagej.util.Images.copy' if available or
        the slower "copy.rai" method. Note that the input RandomAccessibleInterval and
        numpy array must have reversed dimensions relative to each other (e.g. [t, z, y, x, c]
        and [c, x, y, z, t]). Use _permute_rai_to_python() on the RandomAccessibleInterval
        to reorganize the dimensions.

        :param rai: A net.imglib2.RandomAccessibleInterval.
        :param numpy_array: A NumPy array with the same shape as the input RandomAccessibleInterval.
        :return: NumPy array with the input RandomAccessibleInterval data.
        """
        if not isinstance(rai, _RandomAccessibleInterval()):
            raise TypeError("rai is not a RAI")
        if not self._is_arraylike(numpy_array):
            raise TypeError("numpy_array is not arraylike")

        # check imagej-common version for fast copy availability.
        ijc_slow_copy_version = "0.30.0"
        ijc_active_version = sj.get_version(_Dataset())
        fast_copy_available = sj.compare_version(
            ijc_slow_copy_version, ijc_active_version
        )

        if fast_copy_available:
            Images = sj.jimport("net.imagej.util.Images")
            Images.copy(rai, self.to_java(numpy_array))
        else:
            self._ij.op().run("copy.rai", self.to_java(numpy_array), rai)

        return numpy_array

    def run_macro(self, macro: str, args=None):
        """Run an ImageJ macro.

        Run an ImageJ macro by providing the macro code/script in a string and
        the arguments in a dictionary.

        :param macro: The macro code/script as a string.
        :param args: A dictionary of macro arguments in key: valye pairs.
        :return: Runs the specified macro with the given arguments.

        :example:

        .. highlight:: python
        .. code-block:: python

            macro = \"""
            #@ String name
            #@ int age
            output = name + " is " + age " years old."
            \"""
            args = {
                'name': 'Sean',
                'age': 26
            }
            macro_result = ij.py.run_macro(macro, args)
            print(macro_result.getOutput('output'))
        """
        self._ij._check_legacy_active("Use of original ImageJ macros is not possible.")

        try:
            if args is None:
                return self._ij.script().run("macro.ijm", macro, True).get()
            else:
                return (
                    self._ij.script()
                    .run("macro.ijm", macro, True, ij.py.jargs(args))
                    .get()
                )
        except Exception as exc:
            _dump_exception(exc)
            raise exc

    def run_plugin(self, plugin: str, args=[], ij1_style=True, imp=None):
        """Run an ImageJ 1.x plugin.

        Run an ImageJ 1.x plugin by specifying the plugin name as a string,
        and the plugin arguments as a dictionary. For the few plugins that
        use the ImageJ2 style macros (i.e. explicit booleans in the recorder),
        set the option variable ij1_style=False.

        :param plugin: The string name for the plugin command.
        :param args: A dictionary of plugin arguments in key: value pairs.
        :param ij1_style: Boolean to set which implicit boolean style to use (ImageJ or ImageJ2).

        :example:

        .. highlight:: python
        .. code-block:: python

            plugin = 'Mean'
            args = {
                'block_radius_x': 10,
                'block_radius_y': 10
            }
            ij.py.run_plugin(plugin, args)
        """
        self._ij.IJ.run(imp, plugin, self.argstring(args, ij1_style))

    def run_script(self, language: str, script: str, args=None):
        """Run an ImageJ2 script.

        Run a script in one of ImageJ2's supported scripting languages.
        Specify the language of the script, provide the code as a string
        and the arguments as a dictionary.

        :param language: The file extension for the scripting language.
        :param script: A string of the script code.
        :param args: A dictionary of macro arguments in key: value pairs.
        :return: A Java map of output names and values, key: value pais.

        :example:

        .. highlight:: python
        .. code-block:: python

            language = 'ijm'
            script = \"""
            #@ String name
            #@ int age
            output = name + " is " + age " years old."
            \"""
            args = {
                'name': 'Sean',
                'age': 26
            }
            script_result = ij.py.run_script(language, script, args)
            print(script_result.getOutput('output'))
        """
        script_lang = self._ij.script().getLanguageByName(language)
        if script_lang is None:
            script_lang = self._ij.script().getLanguageByExtension(language)
        if script_lang is None:
            raise ValueError("Unknown script language: " + language)
        exts = script_lang.getExtensions()
        if exts.isEmpty():
            raise ValueError(
                "Script language '"
                + script_lang.getLanguageName()
                + "' has no extensions"
            )
        ext = str(exts.get(0))
        try:
            if args is None:
                return self._ij.script().run("script." + ext, script, True).get()
            return (
                self._ij.script()
                .run("script." + ext, script, True, ij.py.jargs(args))
                .get()
            )
        except Exception as exc:
            _dump_exception(exc)
            raise exc

    def show(self, image, cmap=None):
        """Display a Java or Python 2D image.

        Display a Java or Python 2D image.

        :param image: A Java or Python image that can be converted to a NumPy array.
        :param cmap: The colormap for the matplotlib.pyplot image display.
        :return: Displayed image.
        """
        if image is None:
            raise TypeError("Image must not be None")

        # NB: Import this only here on demand, rather than above.
        # Otherwise, some headless systems may experience errors
        # like "ImportError: Failed to import any qt binding".
        from matplotlib import pyplot

        pyplot.imshow(self.from_java(image), interpolation="nearest", cmap=cmap)
        pyplot.show()

    def sync_image(self, imp: "ij.ImagePlus"):
        """Synchronize data between ImageJ and ImageJ2.

        Synchronize between a Dataset or ImageDisplay linked to an
        ImagePlus by accepting the ImagePlus data as true.

        :param imp: The ImagePlus that needs to be synchronized.
        """
        # This code is necessary because an ImagePlus can sometimes be modified without modifying the
        # linked Dataset/ImageDisplay.  This happens when someone uses the ImageProcessor of the ImagePlus to change
        # values on a slice.  The imagej-legacy layer does not synchronize when this happens to prevent
        # significant overhead, as otherwise changing a single pixel would mean syncing a whole slice.  The
        # ImagePlus also has a stack, which in the legacy case links to the Dataset/ImageDisplay.  This stack is
        # updated by the legacy layer when you change slices, using ImageJVirtualStack.setPixelsZeroBasedIndex().
        # As such, we only need to make sure that the current 2D image slice is up to date.  We do this by manually
        # setting the stack to be the same as the imageprocessor.
        stack = imp.getStack()
        pixels = imp.getProcessor().getPixels()
        stack.setPixels(pixels, imp.getCurrentSlice())

    def synchronize_ij1_to_ij2(self, imp: "ij.ImagePlus"):
        """
        This function is deprecated. Use sync_image instead.
        """
        _logger.warning(
            "The synchronize_ij1_to_ij2 function is deprecated. Use sync_image instead."
        )
        self.sync_image(imp)

    def to_dataset(self, data):
        """Convert the data into an ImageJ2 Dataset.

        Converts a Python image (e.g. xarray or numpy array) or Java image (e.g.
        RandomAccessibleInterval or Img) into a 'net.imagej.Dataset' Java object.

        :param data: Image object to be converted to Dataset.
        :return: A 'net.imagej.Dataset'.
        """
        if self._is_xarraylike(data):
            return self._xarray_to_dataset(data)
        if self._is_arraylike(data):
            return self._numpy_to_dataset(data)
        if sj.isjava(data):
            return self._java_to_dataset(data)

        raise TypeError(f"Type not supported: {type(data)}")

    def to_img(self, data):
        """Convert the data into an ImgLib2 Img.

        Converts a Python image (e.g. xarray or numpy array) or Java image (e.g.
        RandomAccessibleInterval) into a 'net.imglib2.img.Img' Java object.

        :param data: Image object to be converted to Img.
        :return: A 'net.imglib2.img.Img'.
        """
        if self._is_xarraylike(data):
            return self._xarray_to_img(data)
        if self._is_arraylike(data):
            return self._numpy_to_img(data)
        if sj.isjava(data):
            return self._java_to_img(data)

        raise TypeError(f"Type not supported: {type(data)}")

    def to_imageplus(self, data):
        """Convert the data into an ImageJ ImagePlus.

        Converts a Python image (e.g. xarray or numpy array) or Java image (e.g.
        RandomAccessibleInterval or Dataset) into an 'ij.ImagePlus' Java object.

        :param data: Image object to be converted to ImagePlus.
        :return: An 'ij.ImagePlus'.
        """
        self._ij._check_legacy_active("Conversion to ImagePlus is not supported.")
        return self._ij.convert().convert(self.to_dataset(data), _ImagePlus())

    def to_java(self, data):
        """Convert supported Python data into Java equivalents.

        Converts Python objects (e.g. 'xarray.DataArray') into the Java
        equivalents. For numpy arrays, the Java image points to the Python array.

        :param data: Python object to be converted into its respective Java counterpart.
        :return: A Java object converted from Python.
        """
        return sj.to_java(data)

    def window_manager(self):
        """
        ij.py.window_manager() is deprecated.
        Use ij.WindowManager instead.
        """
        _logger.warning(
            "ij.py.window_manager() is deprecated. Use ij.WindowManager instead."
        )
        return self._ij.WindowManager

    # -- Helper functions --

    def _assign_dataset_metadata(self, dataset: "net.imagej.Dataset", attrs):
        """
        :param dataset: ImageJ2 Dataset
        :param attrs: Dictionary containing metadata
        """
        dataset.getProperties().putAll(self.to_java(attrs))

    def _format_argument(self, key, value, ij1_style):
        if value is True:
            argument = str(key)
            if not ij1_style:
                argument += "=true"
        elif value is False:
            argument = None
            if not ij1_style:
                argument = f"{key}=false"
        elif value is None:
            raise NotImplementedError("Conversion for None is not yet implemented")
        else:
            val_str = self._format_value(value)
            argument = f"{key}={val_str}"
        return argument

    def _format_value(self, value):
        if isinstance(value, _ImagePlus()):
            return str(value.getTitle())
        temp_value = str(value).replace("\\", "/")
        if temp_value.startswith("[") and temp_value.endswith("]"):
            return temp_value
        final_value = "[" + temp_value + "]"
        return final_value

    def _get_origin(self, axis):
        """
        Get the coordinate origin of an axis, assuming it is the first entry.
        :param axis: A 1D list like entry accessible with indexing, which contains the axis coordinates
        :return: The origin for this axis.
        """
        return axis.values[0]

    def _invert_except_last_element(self, lst):
        """
        Invert a list except for the last element.
        """
        cut_list = lst[0:-1]
        reverse_cut = list(reversed(cut_list))
        reverse_cut.append(lst[-1])
        return reverse_cut

    def _is_arraylike(self, arr):
        return (
            hasattr(arr, "shape")
            and hasattr(arr, "dtype")
            and hasattr(arr, "__array__")
            and hasattr(arr, "ndim")
        )

    def _is_memoryarraylike(self, arr):
        return (
            self._is_arraylike(arr)
            and hasattr(arr, "data")
            and type(arr.data).__name__ == "memoryview"
        )

    def _is_xarraylike(self, xarr):
        return (
            hasattr(xarr, "values")
            and hasattr(xarr, "dims")
            and hasattr(xarr, "coords")
            and self._is_arraylike(xarr.values)
        )

    def _permute_dataset_to_python(self, rai):
        """Wrap a numpy array with xarray and axes metadata from a RandomAccessibleInterval.

        Wraps a numpy array with the metadata from the source RandomAccessibleInterval
        metadata (i.e. axes). Also permutes the dimension of the rai to conform to
        numpy's standards

        :param permuted_rai: A RandomAccessibleInterval with axes (e.g. Dataset or ImgPlus).
        :return: xarray.DataArray with metadata/axes.
        """
        data = self._ij.convert().convert(rai, _ImgPlus())
        permuted_rai = self._permute_rai_to_python(data)
        numpy_result = self.initialize_numpy_image(permuted_rai)
        numpy_result = self.rai_to_numpy(permuted_rai, numpy_result)
        return self._dataset_to_xarray(permuted_rai, numpy_result)

    def _permute_rai_to_python(self, rich_rai: "net.imglib2.RandomAccessibleInterval"):
        """Permute a RandomAccessibleInterval to the python reference order.

        Permute a RandomAccessibleInterval to the Python reference order of
        CXYZT (where dimensions exist). Note that this is reverse from the final array order of
        TZYXC.

        :param rich_rai: A RandomAccessibleInterval with axis labels (e.g. Dataset or ImgPlus).
        :return: A permuted RandomAccessibleInterval.
        """
        # get input rai metadata if it exists
        try:
            rai_metadata = rich_rai.getProperties()
        except AttributeError:
            rai_metadata = None

        axis_types = [axis.type() for axis in rich_rai.dim_axes]

        # permute rai to specified order and transfer metadata
        permute_order = dims.prioritize_rai_axes_order(
            axis_types, dims._python_rai_ref_order()
        )
        permuted_rai = dims.reorganize(rich_rai, permute_order)

        # add metadata to image if it exists
        if rai_metadata != None:
            permuted_rai.getProperties().putAll(rai_metadata)

        return permuted_rai

    # -- Helper functions - type conversion --

    def _add_converters(self):
        """Add all known converters to ScyJava's conversion mechanism."""
        [sj.add_java_converter(c) for c in self._imagej_java_converters()]
        [sj.add_py_converter(c) for c in self._imagej_py_converters()]

    def _imagej_java_converters(self) -> List[sj.Converter]:
        """Get all Python-to-ImgLib2 Converters"""
        return [
            sj.Converter(
                predicate=lambda obj: isinstance(obj, Labeling),
                converter=self._labeling_to_imglabeling,
                priority=sj.Priority.HIGH + 1,
            ),
            sj.Converter(
                predicate=self._is_memoryarraylike,
                converter=self.to_img,
                priority=sj.Priority.HIGH,
            ),
            sj.Converter(
                predicate=self._is_xarraylike,
                converter=self.to_dataset,
                priority=sj.Priority.HIGH + 1,
            ),
        ]

    def _imagej_py_converters(self) -> List[sj.Converter]:
        """Get all ImgLib2-to-Python Converters"""
        return [
            sj.Converter(
                predicate=lambda obj: isinstance(obj, _ImgLabeling()),
                converter=self._imglabeling_to_labeling,
                priority=sj.Priority.HIGH,
            ),
            sj.Converter(
                predicate=lambda obj: _ImagePlus() and isinstance(obj, _ImagePlus()),
                converter=lambda obj: self.from_java(self._imageplus_to_imgplus(obj)),
                priority=sj.Priority.HIGH + 2,
            ),
            sj.Converter(
                predicate=self._can_convert_imgPlus,
                converter=lambda obj: self._permute_dataset_to_python(
                    self._ij.convert().convert(obj, _ImgPlus())
                ),
                priority=sj.Priority.HIGH,
            ),
            sj.Converter(
                predicate=self._can_convert_rai,
                converter=self._convert_rai,
                priority=sj.Priority.HIGH - 2,
            ),
        ]

    def _can_convert_imgPlus(self, obj) -> bool:
        """Return false unless conversion to RAI is possible."""
        try:
            can_convert = self._ij.convert().supports(obj, _ImgPlus())
            has_axis = dims._has_axis(obj)
            return can_convert and has_axis
        except Exception:
            return False

    def _can_convert_rai(self, obj) -> bool:
        """Return false unless conversion to RAI is possible."""
        try:
            return self._ij.convert().supports(obj, _RandomAccessibleInterval())
        except Exception:
            return False

    def _convert_rai(self, data):
        rai = self._ij.convert().convert(data, _RandomAccessibleInterval())
        numpy_result = self.initialize_numpy_image(rai)
        return self.rai_to_numpy(rai, numpy_result)

    def _dataset_to_xarray(
        self, rich_rai: "net.imglib2.RandomAccessibleInterval", numpy_array: np.ndarray
    ) -> xr.DataArray:
        """Wrap a numpy array with xarray and axes metadta from a RandomAccessibleInterval.

        Wraps a numpy array with the metadata from the source RandomAccessibleInterval
        metadata (i.e. axes).

        :param rich_rai: A RandomAccessibleInterval with metadata (e.g. Dataset or ImgPlus).
        :param numpy_array: A np.ndarray to wrap with xarray.
        :return: xarray.DataArray with metadata/axes.
        """
        if not isinstance(rich_rai, _RandomAccessibleInterval()):
            raise TypeError("rich_rai is not a RAI")
        if not hasattr(rich_rai, "dim_axes"):
            raise TypeError("rich_rai is not a rich RAI")
        if not self._is_arraylike(numpy_array):
            raise TypeError("numpy_array is not arraylike")

        # get metadata
        xr_axes = list(rich_rai.dim_axes)
        xr_dims = list(rich_rai.dims)
        xr_attrs = sj.to_python(rich_rai.getProperties())
        # reverse axes and dims to match numpy_array
        xr_axes.reverse()
        xr_dims.reverse()
        xr_dims = dims._convert_dims(xr_dims, direction="python")
        xr_coords = dims._get_axes_coords(xr_axes, xr_dims, numpy_array.shape)
        return xr.DataArray(numpy_array, dims=xr_dims, coords=xr_coords, attrs=xr_attrs)

    def _imageplus_to_imgplus(self, imp):
        if not _ImagePlus() or not isinstance(imp, _ImagePlus()):
            raise ValueError("Input is not an ImagePlus")

        ds = self._ij.convert().convert(imp, _Dataset())
        return ds.getImgPlus()

    def _java_to_dataset(self, data):
        """
        Convert the data into an ImageJ2 Dataset.
        """
        assert sj.isjava(data)
        if isinstance(data, _Dataset()):
            return data

        # NB: This try checking is necessary because the set of ImageJ2 converters is not complete.
        # E.g., there is no way to directly go from Img to Dataset, instead you need to chain the
        # Img->ImgPlus->Dataset converters.
        try:
            if self._ij.convert().supports(data, _Dataset()):
                return self._ij.convert().convert(data, _Dataset())
            if self._ij.convert().supports(data, _ImgPlus()):
                imgPlus = self._ij.convert().convert(data, _ImgPlus())
                return self._ij.dataset().create(imgPlus)
            if self._ij.convert().supports(data, _Img()):
                img = self._ij.convert().convert(data, _Img())
                return self._ij.dataset().create(_ImgPlus()(img))
            if self._ij.convert().supports(data, _RandomAccessibleInterval()):
                rai = self._ij.convert().convert(data, _RandomAccessibleInterval())
                return self._ij.dataset().create(rai)
        except Exception as exc:
            _dump_exception(exc)
            raise exc
        raise TypeError("Cannot convert to dataset: " + str(type(data)))

    def _java_to_img(self, data):
        """
        Convert the data into an ImgLib2 Img.
        """
        assert sj.isjava(data)
        if isinstance(data, _Img()):
            return data

        # NB: This try checking is necessary because the set of ImageJ2 converters is not complete.
        try:
            if self._ij.convert().supports(data, _Img()):
                return self._ij.convert().convert(data, _Img())
            if self._ij.convert().supports(data, _RandomAccessibleInterval()):
                rai = self._ij.convert().convert(data, _RandomAccessibleInterval())
                return _ImgView().wrap(rai)
        except Exception as exc:
            _dump_exception(exc)
            raise exc
        raise TypeError("Cannot convert to img: " + str(type(data)))

    def _numpy_to_dataset(self, data):
        assert self._is_arraylike(data)
        rai = imglyb.to_imglib(data)
        return self._java_to_dataset(rai)

    def _numpy_to_img(self, data):
        assert self._is_arraylike(data)
        rai = imglyb.to_imglib(data)
        return self._java_to_img(rai)

    def _xarray_to_dataset(self, xarr):
        """
        Converts a xarray dataarray to a dataset, inverting C-style (slow axis first) to F-style (slow-axis last)
        :param xarr: Pass an xarray dataarray and turn into a dataset.
        :return: The dataset
        """
        assert self._is_xarraylike(xarr)
        if dims._ends_with_channel_axis(xarr):
            vals = np.moveaxis(xarr.values, -1, 0)
            dataset = self._numpy_to_dataset(vals)
        else:
            dataset = self._numpy_to_dataset(xarr.values)
        axes = dims._assign_axes(xarr)
        dataset.setAxes(axes)
        self._assign_dataset_metadata(dataset, xarr.attrs)

        return dataset

    def _xarray_to_img(self, xarr):
        """
        Converts a xarray dataarray to an img, inverting C-style (slow axis first) to F-style (slow-axis last)
        :param xarr: Pass an xarray dataarray and turn into a img.
        :return: The img
        """
        assert self._is_xarraylike(xarr)
        if dims._ends_with_channel_axis(xarr):
            vals = np.moveaxis(xarr.values, -1, 0)
            return self._numpy_to_img(vals)
        else:
            return self._numpy_to_img(xarr.values)

    # -- Helper functions - labelings --

    def _delete_labeling_files(self, filepath):
        """
        Removes any Labeling data left over at filepath
        :param filepath: the filepath where Labeling (might have) saved data
        """
        pth_json = filepath + ".lbl.json"
        pth_tif = filepath + ".tif"
        if os.path.exists(pth_tif):
            os.remove(pth_tif)
        if os.path.exists(pth_json):
            os.remove(pth_json)

    def _imglabeling_to_labeling(self, data):
        """
        Converts an ImgLabeling to an equivalent Python Labeling
        :param data: the data
        :return: a Labeling
        """
        LabelingIOService = sj.jimport("io.scif.labeling.LabelingIOService")
        labels = self._ij.context().getService(LabelingIOService)

        # Save the image on the java side
        tmp_pth = os.getcwd() + "/tmp"
        tmp_pth_json = tmp_pth + ".lbl.json"
        tmp_pth_tif = tmp_pth + ".tif"
        try:
            self._delete_labeling_files(tmp_pth)
            data = self._ij.convert().convert(data, _ImgLabeling())
            labels.save(
                data, tmp_pth_tif
            )  # TODO: improve, likely utilizing the data's name
        except JException:
            print("Failed to save the data")

        # Load the labeling on the python side
        labeling = Labeling.from_file(tmp_pth_json)
        self._delete_labeling_files(tmp_pth)
        return labeling

    def _labeling_to_imglabeling(self, data):
        """
        Converts a python Labeling to an equivalent ImgLabeling
        :param data: the data
        :return: an ImgLabeling
        """
        LabelingIOService = sj.jimport("io.scif.labeling.LabelingIOService")
        labels = self._ij.context().getService(LabelingIOService)

        # Save the image on the python side
        tmp_pth = "./tmp"
        self._delete_labeling_files(tmp_pth)
        data.save_result(tmp_pth)

        # Load the labeling on the python side
        try:
            tmp_pth_json = tmp_pth + ".lbl.json"
            labeling = labels.load(tmp_pth_json, JObject, JObject)
        except JException as exc:
            self._delete_labeling_files(tmp_pth)
            raise exc
        self._delete_labeling_files(tmp_pth)

        return labeling


@JImplementationFor("net.imagej.ImageJ")
class GatewayAddons(object):
    """ImageJ2 gateway addons.

    This class should not be initialized manually. Upon initialization
    the GatewayAddons class is attached to the newly initialized
    ImageJ2 instance via JPype's class customization mechanism:

    https://jpype.readthedocs.io/en/latest/userguide.html#class-customizers
    """

    @property
    @lru_cache(maxsize=None)
    def py(self):
        """Access the ImageJPython convenience methods.

        Access to the ImageJPython convenience methods through
        the '.py' attribute.

        :return: ImageJPython convenience methods.
        """
        return ImageJPython(self)

    @property
    def legacy(self):
        """Get the ImageJ2 LegacyService.

        Gets the ImageJ2 gateway's LegacyService, or None if original
        ImageJ support is not available in the current environment.

        :return: The ImageJ2 LegacyService.
        """
        if not hasattr(self, "_legacy"):
            try:
                LegacyService = sj.jimport("net.imagej.legacy.LegacyService")
                self._legacy = self.get("net.imagej.legacy.LegacyService")
                if self.ui().isHeadless():
                    _logger.warning(
                        "Operating in headless mode - the original ImageJ will have limited functionality."
                    )
            except TypeError:
                self._legacy = None

        return self._legacy

    @property
    def IJ(self):
        """Get the original ImageJ `IJ` utility class.

        :return: The `ij.IJ` class.
        """
        return self._access_legacy_class("ij.IJ")

    @property
    def ResultsTable(self):
        """Get the original ImageJ `ResultsTable` class.

        :return: The `ij.measure.ResultsTable` class.
        """
        return self._access_legacy_class("ij.measure.ResultsTable")

    @property
    def RoiManager(self):
        """Get the original ImageJ `RoiManager` class.

        :return: The `ij.plugin.frame.RoiManager` class.
        """
        return self._access_legacy_class("ij.plugin.frame.RoiManager")

    @property
    def WindowManager(self):
        """Get the original ImageJ `WindowManager` class.

        :return: The `ij.WindowManager` class.
        """
        return self._access_legacy_class("ij.WindowManager")

    def _access_legacy_class(self, fqcn: str):
        self._check_legacy_active(f"The {fqcn} class is not available.")
        class_name = fqcn[fqcn.rindex(".") + 1 :]
        property_name = f"_{class_name}"
        if not hasattr(self, property_name):
            if self.ui().isHeadless():
                _logger.warning(
                    f"Operating in headless mode - the {class_name} class will not be fully functional."
                )
            setattr(self, property_name, sj.jimport(fqcn))

        return getattr(self, property_name)

    def _check_legacy_active(self, usage_context=""):
        if not self.legacy or not self.legacy.isActive():
            raise ImportError(
                f"The original ImageJ is not available in this environment. {usage_context} See: https://github.com/imagej/pyimagej/blob/master/doc/Initialization.md"
            )


@JImplementationFor("net.imglib2.EuclideanSpace")
class EuclideanSpaceAddons(object):
    @property
    def ndim(self):
        """Get the number of dimensions.

        :return: Number of dimensions.
        :see: net.imglib2.EuclideanSpace#numDimensions()
        """
        return self.numDimensions()


@JImplementationFor("net.imglib2.Interval")
class IntervalAddons(object):
    @property
    def shape(self):
        """Get the shape of the interval.

        :return: Tuple of the interval shape.
        :see: net.imglib2.Interval#dimension(int)
        """
        return tuple(self.dimension(d) for d in range(self.numDimensions()))


@JImplementationFor("net.imglib2.RandomAccessibleInterval")
class RAIOperators(object):
    """RandomAccessibleInterval operators.

    This class should not be initialized manually. Upon initialization
    the RAIOperators class automatically extends the Java
    RandomAccessibleInterval via JPype's class customization mechanism:

    https://jpype.readthedocs.io/en/latest/userguide.html#class-customizers
    """

    def __add__(self, other):
        """Return self + value."""
        return (
            self._op.run("math.add", self, other)
            if self._op is not None
            else self._ImgMath(self, other, "add")
        )

    def __sub__(self, other):
        """Return self - value."""
        return (
            self._op.run("math.sub", self, other)
            if self._op is not None
            else self._ImgMath(self, other, "sub")
        )

    def __mul__(self, other):
        """Return self * value."""
        return (
            self._op.run("math.mul", self, other)
            if self._op is not None
            else self._ImgMath(self, other, "mul")
        )

    def __truediv__(self, other):
        """Return self / value."""
        return (
            self._op.run("math.div", self, other)
            if self._op is not None
            else self._ImgMath(self, other, "div")
        )

    def __getitem__(self, key):
        if type(key) == slice:
            # Wrap single slice into tuple of length 1.
            return self._slice((key,))
        elif type(key) == tuple:
            return self._index(key) if self._is_index(key) else self._slice(key)
        elif type(key) == int:
            # Wrap single int into tuple of length 1.
            return self.__getitem__((key,))
        else:
            raise ValueError(f"Invalid key type: {type(key)}")

    @property
    def dtype(self):
        """Get the dtype of a RandomAccessibleInterval,
        a subclass of net.imglib2.type.Type.

        :return: dtype of the RandomAccessibleInterval.
        """
        Util = sj.jimport("net.imglib2.util.Util")
        return type(Util.getTypeFromInterval(self))

    def squeeze(self, axis=None):
        """Remove axes of length one from array.

        :return: Squeezed RandomAccessibleInterval.
        """
        if axis is None:
            # Process all dimensions.
            axis = tuple(range(self.numDimensions()))
        if type(axis) == int:
            # Convert int to singleton tuple.
            axis = (axis,)
        if type(axis) != tuple:
            raise ValueError(f"Invalid type for axis parameter: {type(axis)}")

        Views = sj.jimport("net.imglib2.view.Views")
        res = self
        for d in range(self.numDimensions() - 1, -1, -1):
            if d in axis and self.dimension(d) == 1:
                res = Views.hyperSlice(res, d, self.min(d))
        return res

    @property
    def T(self):
        """Transpose RandomAccessibleInterval.

        :return: Transposed RandomAccessibleInterval.
        """
        return self.transpose

    @property
    def transpose(self):
        """Transpose RandomAccessibleInterval.

        :return: Transposed RandomAccessibleInterval.
        """
        view = self
        max_dim = self.numDimensions() - 1
        for i in range(self.numDimensions() // 2):
            if self._op is not None:
                view = ij.op().run("transform.permuteView", self, i, max_dim - i)
            else:
                raise RuntimeError(f"OpService is unavailable for this operation.")
        return view

    def _index(self, position):
        ra = self._ra
        # Can we store this as a shape property?
        if stack._index_within_range(position, self.shape):
            for i in range(len(position)):
                pos = position[i]
                if pos < 0:
                    pos += self.shape[i]
                ra.setPosition(pos, i)
            return ra.get()

    def _ImgMath(self, other, operation: str):
        ImgMath = sj.jimport("net.imglib2.algorithm.math.ImgMath")
        ImgMath_operations = {
            "add": ImgMath.add(self._jargs(self, other)),
            "sub": ImgMath.sub(self._jargs(self, other)),
            "mul": ImgMath.mul(self._jargs(self, other)),
            "div": ImgMath.div(self._jargs(self, other)),
        }
        return ImgMath_operations[operation]

    def _is_index(self, a):
        # Check dimensionality - if we don't have enough dims, it's a slice
        num_dims = 1 if type(a) == int else len(a)
        if num_dims < self.numDimensions():
            return False
        # if an int, it is an index
        if type(a) == int:
            return True
        # if we have a tuple, it's an index if there are any slices
        hasSlice = True in [type(item) == slice for item in a]
        return not hasSlice

    def _jargs(self, *args):
        return _JObjectArray()(list(map(sj.to_java, args)))

    @property
    @lru_cache(maxsize=None)
    def _op(self):
        # check if has getcontext() attribute
        op = None
        if hasattr(self, "getContext"):
            op = self.getContext().getService("net.imagej.ops.OpService")

        # if not context, try to get global ij or return None
        if op == None:
            try:
                return ij.op()
            except:
                return None
        else:
            return op

    @property
    def _ra(self):
        threadLocal = getattr(self, "_threadLocal", None)
        if threadLocal is None:
            with rai_lock:
                threadLocal = getattr(self, "_threadLocal", None)
                if threadLocal is None:
                    threadLocal = threading.local()
                    self._threadLocal = threadLocal
        ra = getattr(threadLocal, "ra", None)
        if ra is None:
            with rai_lock:
                ra = getattr(threadLocal, "ra", None)
                if ra is None:
                    ra = self.randomAccess()
                    threadLocal.ra = ra
        return ra

    def _slice(self, ranges):
        expected_dims = len(ranges)
        actual_dims = self.numDimensions()
        if expected_dims > actual_dims:
            raise ValueError(f"Dimension mismatch: {expected_dims} > {actual_dims}")
        elif expected_dims < actual_dims:
            ranges = (list(ranges) + actual_dims * [slice(None)])[:actual_dims]
        imin = []
        imax = []
        istep = []
        dslices = [r if type(r) == slice else slice(r, r + 1) for r in ranges]
        for dslice in dslices:
            imax.append(None if dslice.stop == None else dslice.stop - 1)
            imin.append(None if dslice.start == None else dslice.start)
            istep.append(1 if dslice.step == None else dslice.step)

        # BE WARNED! This does not yet preserve net.imagej-level axis metadata!
        # We need to finish RichImg to support that properly.

        return stack.rai_slice(self, tuple(imin), tuple(imax), tuple(istep))


@JImplementationFor("net.imagej.space.TypedSpace")
class TypedSpaceAddons(object):
    """TypedSpace addons.

    This class should not be initialized manually. Upon initialization
    the TypedSpaceAddons class automatically extends the Java
    net.imagej.space.TypedSpace via JPype's class customization mechanism:

    https://jpype.readthedocs.io/en/latest/userguide.html#class-customizers
    """

    @property
    def dims(self) -> Tuple[str]:
        """Get the axis labels of the dimensional space.

        :return: Dimension labels of the space.
        :see: net.imagej.space.TypedSpace#axis(int)
        """
        return tuple(str(axis.type()) for axis in self.dim_axes)


@JImplementationFor("net.imagej.space.AnnotatedSpace")
class AnnotatedSpaceAddons(object):
    """AnnotatedSpace addons.

    This class should not be initialized manually. Upon initialization
    the AnnotatedSpaceAddons class automatically extends the Java
    net.imagej.space.AnnotatedSpace via JPype's class customization mechanism:

    https://jpype.readthedocs.io/en/latest/userguide.html#class-customizers
    """

    @property
    def dim_axes(self) -> Tuple["net.imagej.axis.Axis"]:
        """Get the axes of the dimensional space.

        :return: tuple of net.imagej.axis.Axis objects describing the
                 dimensional axes.
        :see: net.imagej.space.AnnotatedSpace#axis(int)
        """
        return tuple(self.axis(d) for d in range(self.numDimensions()))


@JImplementationFor("ij.ImagePlus")
class ImagePlusAddons(object):
    """ImagePlus addons.

    This class should not be initialized manually. Upon initialization
    the ImagePlusAddons class automatically extends the Java
    ij.ImagePlus via JPype's class customization mechanism:

    https://jpype.readthedocs.io/en/latest/userguide.html#class-customizers
    """

    @property
    def dims(self) -> Tuple[str]:
        """Get the dimensional axis labels of the image.

        ImagePlus objects are always ordered XYZCT, although
        this function squeezes out dimensions of length 1.

        :return: Dimension labels of the image.
        """
        return tuple(
            "XYCZT"[d] for d, length in enumerate(self.getDimensions()) if length > 1
        )

    @property
    def shape(self):
        """Get the shape of the image.

        :return: Tuple of the image shape.
        :see: ij.ImagePlus#getDimensions()
        """
        return tuple(length for length in self.getDimensions() if length > 1)


def init(
    ij_dir_or_version_or_endpoint=None,
    mode=Mode.HEADLESS,
    add_legacy=True,
    headless=None,
):
    """Initialize an ImageJ2 environment.

    The environment can wrap a local ImageJ2 installation, or consist of a
    specific version of ImageJ2 downloaded on demand, or even an explicit list
    of Maven artifacts. The environment can be initialized in headless mode or
    GUI mode, and with or without support for the original ImageJ.
    Note: some original ImageJ operations do not function in headless mode.

    :param ij_dir_or_version_or_endpoint:

        | Path to a local ImageJ2 installation (e.g. /Applications/Fiji.app),
        | OR version of net.imagej:imagej artifact to launch (e.g. 2.3.0),
        | OR endpoint of another artifact built on ImageJ2 (e.g. sc.fiji:fiji),
        | OR list of Maven artifacts to include (e.g.
        |   ['net.imagej:imagej:2.3.0', 'net.imagej:imagej-legacy', 'net.preibisch:BigStitcher']).
        | The default is the latest version of net.imagej:imagej.

    :param mode:

        How the environment will behave. Options include:

        * Mode.HEADLESS -
            Start the JVM in headless mode, i.e. with no GUI. This is the
            default mode. Useful if you want to use ImageJ as a library, or
            run it on a remote server.
            NB: In this mode with add_legacy=True, not all functions of the
            original ImageJ are available; in particular, some plugins do
            not work properly because they assume ImageJ has a GUI.
        * Mode.GUI -
            Start ImageJ2 as a GUI application, displaying the GUI
            automatically and then blocking.
            NB: In this mode with add_legacy=True, the JVM and Python will
            both terminate when ImageJ closes!
        * Mode.INTERACTIVE -
            Start ImageJ2 with GUI support, but *not* displaying the GUI
            automatically, Does not block. To display the GUI in this mode,
            call ij.ui().showUI().
            NB: This mode is not available on macOS, due to its application
            threading model.
            NB: In this mode with add_legacy=True, the JVM and Python will
            both terminate when ImageJ closes!

    :param add_legacy:

        Whether or not to include support for original ImageJ functionality.
        If True, original ImageJ functions (ij.* packages) will be available.
        If False, the environment will be "pure ImageJ2", without ij.* support.
        NB: With legacy support enabled in GUI or interactive mode,
        the JVM and Python will both terminate when ImageJ closes!
        For further details, see: https://imagej.net/libs/imagej-legacy

    :param headless:

        Deprecated. Please use the mode parameter instead.

    :return: An instance of the net.imagej.ImageJ gateway

    :example:

    .. highlight:: python
    .. code-block:: python

        ij = imagej.init('sc.fiji:fiji', mode=imagej.Mode.GUI)
    """
    if headless is not None:
        _logger.warning(
            "The headless flag of imagej.init is deprecated. Use the mode argument instead."
        )
        mode = Mode.HEADLESS if headless else Mode.INTERACTIVE

    macos = sys.platform == "darwin"

    if macos and mode == Mode.INTERACTIVE:
        raise EnvironmentError("Sorry, the interactive mode is not available on macOS.")

    if not sj.jvm_started():
        success = _create_jvm(ij_dir_or_version_or_endpoint, mode, add_legacy)
        if not success:
            raise RuntimeError("Failed to create a JVM with the requested environment.")

    if mode == Mode.GUI:
        # Show the GUI and block.
        if macos:
            # NB: This will block the calling (main) thread forever!
            try:
                setupGuiEnvironment(lambda: _create_gateway().ui().showUI())
            except ModuleNotFoundError as e:
                if e.msg == "No module named 'PyObjCTools'":
                    advice = (
                        "PyObjC is required for GUI mode on macOS. Please install it.\n"
                    )
                    if "CONDA_PREFIX" in os.environ:
                        advice += "E.g.: conda install -c conda-forge pyobjc-core pyobjc-framework-cocoa"
                    else:
                        advice += "E.g.: pip install pyobjc"
                    raise RuntimeError(
                        f"Failed to set up macOS GUI environment.\n{advice}"
                    )
                else:
                    raise
        else:
            # Create and show the application.
            gateway = _create_gateway()
            gateway.ui().showUI()
            # We are responsible for our own blocking.
            # TODO: Poll using something better than ui().isVisible().
            while gateway.ui().isVisible():
                time.sleep(1)
            return None
    else:
        # HEADLESS or INTERACTIVE mode: create the gateway and return it.
        return _create_gateway()


def imagej_main():
    """Entry point for launching ImageJ from the command line via the `imagej` console entry point script."""
    args = []
    for i in range(1, len(sys.argv)):
        args.append(sys.argv[i])
    mode = "headless" if "--headless" in args else "gui"
    ij = init(mode=mode)


def _create_gateway():
    # Initialize ImageJ2
    try:
        ImageJ = sj.jimport("net.imagej.ImageJ")
    except TypeError:
        _logger.error(
            """
***Invalid initialization: ImageJ2 was not found***
   Please update your initialization call to include an ImageJ2 application or endpoint (e.g. net.imagej:imagej).
   NOTE: You MUST restart your python interpreter as Java can only be started once.
"""
        )
        return False

    global ij
    ij = ImageJ()

    # Forward stdout and stderr from Java to Python.
    @JImplements("org.scijava.console.OutputListener")
    class JavaOutputListener:
        @JOverride
        def outputOccurred(self, e):
            source = e.getSource().toString
            output = e.getOutput()
            if source == "STDOUT":
                sys.stdout.write(output)
            elif source == "STDERR":
                sys.stderr.write(output)
            else:
                sys.stderr.write(f"[{source}] {output}")

    ij.py._outputMapper = JavaOutputListener()
    ij.console().addOutputListener(ij.py._outputMapper)

    sj.when_jvm_stops(lambda: ij.dispose())

    return ij


def _create_jvm(
    ij_dir_or_version_or_endpoint=None, mode=Mode.HEADLESS, add_legacy=True
):
    """
    Ensures the JVM is properly initialized and ready to go,
    with requested settings.

    :return: True iff the JVM was successfully started.
             Note that this function returns False if a JVM is already running;
             to check for that situation, you can use scyjava.jvm_started().
    """

    # Check if JPype JVM is already running
    if sj.jvm_started():
        _logger.debug("The JVM is already running.")
        return False

    # Initialize configuration.
    if mode == Mode.HEADLESS:
        sj.config.add_option("-Djava.awt.headless=true")
    try:
        if hasattr(sj, "jvm_version") and sj.jvm_version()[0] >= 9:
            # Disable illegal reflection access warnings.
            sj.config.add_option("--add-opens=java.base/java.lang=ALL-UNNAMED")
            sj.config.add_option("--add-opens=java.base/java.util=ALL-UNNAMED")
    except RuntimeError as e:
        _logger.warning("Failed to guess the Java version.")
        _logger.debug(e, exc_info=True)

    # We want ImageJ2's endpoints to come first, so these will be restored later
    original_endpoints = sj.config.endpoints.copy()
    sj.config.endpoints.clear()
    init_failed = False

    if ij_dir_or_version_or_endpoint is None:
        # Use latest release of ImageJ2.
        _logger.debug("Using newest ImageJ2 release")
        sj.config.endpoints.append("net.imagej:imagej")

    elif isinstance(ij_dir_or_version_or_endpoint, list):
        # Assume that this is a list of Maven endpoints
        if any(
            item.startswith("net.imagej:imagej-legacy")
            for item in ij_dir_or_version_or_endpoint
        ):
            add_legacy = False

        endpoint = "+".join(ij_dir_or_version_or_endpoint)
        _logger.debug(
            "List of Maven coordinates given: %s", ij_dir_or_version_or_endpoint
        )
        sj.config.endpoints.append(endpoint)

    elif os.path.isdir(ij_dir_or_version_or_endpoint):
        # Assume path to local ImageJ2 installation.
        add_legacy = False
        path = os.path.abspath(ij_dir_or_version_or_endpoint)
        _logger.debug("Local path to ImageJ2 installation given: %s", path)
        num_jars = _set_ij_env(path)
        if num_jars <= 0:
            _logger.error(
                "Given directory does not appear to be a valid ImageJ2 installation: %s",
                path,
            )
            init_failed = True
        else:
            _logger.info("Added " + str(num_jars + 1) + " JARs to the Java classpath.")
            plugins_dir = str(Path(path) / "plugins")
            sj.config.add_option("-Dplugins.dir=" + plugins_dir)

            # All is well -- now adjust the CWD to the ImageJ2 app directory.
            # See https://github.com/imagej/pyimagej/issues/150.
            os.chdir(path)

    elif re.match("^(/|[A-Za-z]:)", ij_dir_or_version_or_endpoint):
        # Looks like a file path was intended, but it's not a folder.
        path = ij_dir_or_version_or_endpoint
        _logger.error("Local path given is not a directory: %s", path)
        init_failed = True

    elif ":" in ij_dir_or_version_or_endpoint:
        # Assume endpoint of an artifact.
        # Strip out white spaces
        endpoint = ij_dir_or_version_or_endpoint.replace("    ", "")
        if any(
            item.startswith("net.imagej:imagej-legacy") for item in endpoint.split("+")
        ):
            add_legacy = False
        _logger.debug("Maven coordinate given: %s", endpoint)
        sj.config.endpoints.append(endpoint)

    else:
        # Assume version of net.imagej:imagej.
        version = ij_dir_or_version_or_endpoint
        # Skip ignore
        if not re.match("\\d+\\.\\d+\\.\\d+", version):
            _logger.error("Invalid initialization string: %s", version)
            init_failed = True
            return False
        else:
            _logger.debug("ImageJ2 version given: %s", version)
            sj.config.endpoints.append("net.imagej:imagej:" + version)

    if init_failed:
        # Restore any pre-existing endpoints to allow for re-initialization
        sj.config.endpoints.clear()
        sj.config.endpoints.extend(original_endpoints)
        return False

    if add_legacy:
        sj.config.endpoints.append("net.imagej:imagej-legacy:MANAGED")

    # Add additional ImageJ endpoints specific to PyImageJ
    sj.config.endpoints.append("io.scif:scifio-labeling:0.3.1")

    # Restore any pre-existing endpoints, after ImageJ2's
    sj.config.endpoints.extend(original_endpoints)

    try:
        sj.start_jvm()
    except subprocess.CalledProcessError as e:
        # Check to see if initialization failed due to "un-managed" imagej-legacy
        err_lines = []
        unmanaged_legacy = False
        if e.stdout:
            err_lines += e.stdout.decode().splitlines()
        if e.stderr:
            err_lines += e.stderr.decode().splitlines()
        for l in err_lines:
            if (
                "'dependencies.dependency.version' for net.imagej:imagej-legacy:jar is missing."
                in l
            ):
                unmanaged_legacy = True
        if unmanaged_legacy:
            _logger.error(
                """
***Invalid Initialization: you may be using primary endpoint that lacks pom-scijava as a parent***
   To keep all Java components at compatible versions we recommend using a primary endpoint with a pom-scijava parent.
   For example, by putting 'net.imagej:imagej' first in your list of endpoints.
   If you are sure you DO NOT want a primary endpoint with a pom-scijava parent, please re-initialize with 'add_legacy=False'.
"""
            )
        sj.config.endpoints.clear()
        sj.config.endpoints.extend(original_endpoints)
        return False

    return True


def _dump_exception(exc):
    if _logger.isEnabledFor(logging.DEBUG):
        jtrace = jstacktrace(exc)
        if jtrace:
            _logger.debug(jtrace)


def _search_for_jars(target_dir, subfolder=""):
    """
    Search and recursively add .jar files to a list.

    :param target_dir: Base path to search.
    :param subfolder: Optional sub-directory to start the search.
    :return: A list of jar files.
    """
    jars = []
    for root, dirs, files in os.walk(target_dir + subfolder):
        for f in files:
            if f.endswith(".jar"):
                path = root + "/" + f
                jars.append(path)
                _logger.debug("Added %s", path)
    return jars


def _set_ij_env(ij_dir):
    """
    Create a list of required jars and add to the java classpath.

    :param ij_dir: System path for Fiji.app.
    :return: num_jar(int): Number of jars added.
    """
    jars = []
    # search jars directory
    jars.extend(_search_for_jars(ij_dir, "/jars"))
    # search plugins directory
    jars.extend(_search_for_jars(ij_dir, "/plugins"))
    # add to classpath
    sj.config.add_classpath(os.pathsep.join(jars))
    return len(jars)


# Import Java resources on demand.
@lru_cache(maxsize=None)
def _Dataset():
    return sj.jimport("net.imagej.Dataset")


@lru_cache(maxsize=None)
def _ImagePlus():
    try:
        return sj.jimport("ij.ImagePlus")
    except TypeError:
        # No original ImageJ on the classpath.
        return None


@lru_cache(maxsize=None)
def _Img():
    return sj.jimport("net.imglib2.img.Img")


@lru_cache(maxsize=None)
def _ImgLabeling():
    return sj.jimport("net.imglib2.roi.labeling.ImgLabeling")


@lru_cache(maxsize=None)
def _ImgPlus():
    return sj.jimport("net.imagej.ImgPlus")


@lru_cache(maxsize=None)
def _ImgView():
    return sj.jimport("net.imglib2.img.ImgView")


@lru_cache(maxsize=None)
def _RandomAccessibleInterval():
    return sj.jimport("net.imglib2.RandomAccessibleInterval")


@lru_cache(maxsize=None)
def _JObjectArray():
    return JArray(JObject)
