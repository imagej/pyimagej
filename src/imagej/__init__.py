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
    image_url = "https://imagej.net/images/clown.png"
    jimage = ij.io().open(image_url)

    # Convert the image from ImageJ2 to xarray, a package that adds
    # labeled datasets to numpy (http://xarray.pydata.org/en/stable/).
    image = ij.py.from_java(jimage)

    # Display the image (backed by matplotlib).
    ij.py.show(image, cmap="gray")
"""

import logging
import os
import re
import subprocess
import sys
import threading
import time
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import scyjava as sj
import xarray as xr
from jpype import JImplementationFor, setupGuiEnvironment
from scyjava.config import find_jars

import imagej.convert as convert
import imagej.dims as dims
import imagej.images as images
import imagej.stack as stack
from imagej._java import JObjectArray, jc
from imagej._java import log_exception as _log_exception

__author__ = "ImageJ2 developers"
__version__ = sj.get_version("pyimagej")

_logger = logging.getLogger(__name__)
rai_lock = threading.Lock()

# Enable debug logging if DEBUG environment variable is set.
try:
    debug = os.environ["DEBUG"]
    if debug:
        _logger.setLevel(logging.DEBUG)
        dims._logger.setLevel(logging.DEBUG)
except KeyError:
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

    def active_dataset(self) -> "jc.Dataset":
        """Get the active Dataset image.

        Get the active image as a Dataset from the Dataset service.

        :return: The Dataset corresponding to the active image.
        """
        return self._ij.imageDisplay().getActiveDataset()

    def active_imageplus(self, sync: bool = True) -> "jc.ImagePlus":
        """Get the active ImagePlus image.

        Get the active image as an ImagePlus, optionally synchronizing from ImageJ
        to ImageJ2.

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

        Get the active image as an xarray.DataArray, synchronizing from ImageJ
        to ImageJ2.

        :param sync: Synchronize the current ImagePlus slice if True.
        :return: xarray.DataArray array containing the image data.
        """
        # todo: make the behavior use pure ImageJ2 if legacy is not active

        if self._ij.legacy and self._ij.legacy.isActive():
            imp = self.active_imageplus(sync=sync)
            return self.from_java(imp)
        else:
            dataset = self.active_dataset()
            return self.from_java(dataset)

    def argstring(self, args, ij1_style=True):
        """
        Assemble an ImageJ (1.x) argument string from arguments in a dict.

        :param args: A dict of arguments in key/value pairs
        :param ij1_style: True to use implicit booleans in original ImageJ style, or
            False for explicit booleans in ImageJ2 style
        :return: A string version of the arguments
        """
        if isinstance(args, str):
            return args
        else:
            formatted_args = []
            for key, value in args.items():
                arg = self._format_argument(key, value, ij1_style)
                if arg is not None:
                    formatted_args.append(arg)
            return " ".join(formatted_args)

    def dtype(self, image_or_type):
        """Get the dtype of the input image as a numpy.dtype object.

        Note: for Java-based images, this is different than the image's dtype
        property, because ImgLib2-based images report their dtype as a subclass
        of net.imglib2.type.Type, and ImagePlus images do not yet implement
        the dtype function (see https://github.com/imagej/pyimagej/issues/194).

        :param image_or_type:
            | A NumPy array.
            | OR A NumPy array dtype.
            | OR An ImgLib2 image (net.imglib2.Interval).
            | OR An ImageJ2 Dataset (net.imagej.Dataset).
            | OR An ImageJ ImagePlus (ij.ImagePlus).

        :return: Input image dtype.
        """
        return images.dtype(image_or_type)

    def from_java(self, data):
        """Convert supported Java data into Python equivalents.

        Converts Java objects (e.g. net.imagej.Dataset) into the Python
        equivalents.

        :param data: Java object to be converted into its respective Python counterpart.
        :return: A Python object converted from Java.
        """
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
        return images.create_ndarray(image)

    def jargs(self, *args):
        """Convert Python arguments into a Java Object[]

        Converts Python arguments into a Java Object[] (i.e.: array of Java
        objects). This is particularly useful in combination with ImageJ2's
        various run functions, including ij.command().run(...),
        ij.module().run(...), ij.script().run(...), and ij.op().run(...).

        :param args: The Python arguments to wrap into an Object[].
        :return: A Java Object[]
        """
        return JObjectArray()([self.to_java(arg) for arg in args])

    def rai_to_numpy(
        self, rai: "jc.RandomAccessibleInterval", numpy_array: np.ndarray
    ) -> np.ndarray:
        """Copy a RandomAccessibleInterval into a numpy array.

        The input RandomAccessibleInterval is copied into the pre-initialized numpy
        array with either "fast copy" via net.imagej.util.Images.copy if available or
        the slower "copy.rai" method. Note that the input RandomAccessibleInterval and
        numpy array must have reversed dimensions relative to each other
        (e.g. ["t", "z", "y", "x", "c"] and ["c", "x", "y", "z", "t"]).

        :param rai: A net.imglib2.RandomAccessibleInterval.
        :param numpy_array: A NumPy array with the same shape as the input
            RandomAccessibleInterval.
        :return: NumPy array with the input RandomAccessibleInterval data.
        """
        images.copy_rai_into_ndarray(self._ij, rai, numpy_array)
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
                "name": "Sean",
                "age": 26
            }
            macro_result = ij.py.run_macro(macro, args)
            print(macro_result.getOutput("output"))
        """
        self._ij._check_legacy_active("Use of original ImageJ macros is not possible.")

        try:
            if args is None:
                return self._ij.script().run("macro.ijm", macro, True).get()
            else:
                return (
                    self._ij.script()
                    .run("macro.ijm", macro, True, self._ij.py.jargs(args))
                    .get()
                )
        except Exception as exc:
            _log_exception(_logger, exc)
            raise exc

    def run_plugin(
        self, plugin: str, args=None, ij1_style: bool = True, imp: "jc.ImagePlus" = None
    ):
        """Run an ImageJ 1.x plugin.

        Run an ImageJ 1.x plugin by specifying the plugin name as a string,
        and the plugin arguments as a dictionary. For the few plugins that
        use the ImageJ2 style macros (i.e. explicit booleans in the recorder),
        set the option variable ij1_style=False.

        :param plugin: The string name for the plugin command.
        :param args: A dictionary of plugin arguments in key: value pairs.
        :param ij1_style: Boolean to set which implicit boolean style to use
            (ImageJ or ImageJ2).
        :param imp: Optionally: the image to pass to the plugin execution.

        :example:

        .. highlight:: python
        .. code-block:: python

            plugin = "Mean"
            args = {
                "block_radius_x": 10,
                "block_radius_y": 10
            }
            ij.py.run_plugin(plugin, args)
        """
        if args is None:
            args = {}
        argline = self.argstring(args, ij1_style)
        if imp is None:
            # NB: Avoid ambiguous overload between:
            # - IJ.run(ij.ImagePlus, String, String)
            # - IJ.run(ij.macro.Interpreter, String, String)
            self._ij.IJ.run(plugin, argline)
        else:
            self._ij.IJ.run(imp, plugin, argline)

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

            language = "ijm"
            script = \"""
            #@ String name
            #@ int age
            output = name + " is " + age " years old."
            \"""
            args = {
                "name": "Sean",
                "age": 26
            }
            script_result = ij.py.run_script(language, script, args)
            print(script_result.getOutput("output"))
        """
        script_lang = self._ij.script().getLanguageByName(language)
        if script_lang is None:
            script_lang = self._ij.script().getLanguageByExtension(language)
        if script_lang is None:
            raise ValueError("Unknown script language: " + language)
        exts = script_lang.getExtensions()
        if exts.isEmpty():
            raise ValueError(
                f"Script language '{script_lang.getLanguageName()}' has no extensions"
            )
        ext = str(exts.get(0))
        try:
            if args is None:
                return self._ij.script().run("script." + ext, script, True).get()
            return (
                self._ij.script()
                .run("script." + ext, script, True, self._ij.py.jargs(args))
                .get()
            )
        except Exception as exc:
            _log_exception(_logger, exc)
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

    def sync_image(self, imp: "jc.ImagePlus" = None):
        """Synchronize data between ImageJ and ImageJ2.

        Synchronize between a Dataset or ImageDisplay linked to an
        ImagePlus by accepting the ImagePlus data as true.

        :param imp: The ImagePlus that needs to be synchronized,
                    or None to synchronize ImageJ's active image.
        """
        # This code is necessary because an ImagePlus can sometimes be modified without
        # modifying the linked Dataset/ImageDisplay. This happens when someone uses
        # the ImageProcessor of the ImagePlus to change values on a slice. The
        # imagej-legacy layer does not synchronize when this happens to prevent
        # significant overhead, as otherwise changing a single pixel would mean syncing
        # a whole slice. The ImagePlus also has a stack, which in the legacy case links
        # to the Dataset/ImageDisplay. This stack is updated by the legacy layer when
        # you change slices, using ImageJVirtualStack.setPixelsZeroBasedIndex(). As
        # such, we only need to make sure that the current 2D image slice is up to
        # date. We do this by manually setting the stack to be the same as the
        # imageprocessor.
        if imp is None and self._ij.legacy and self._ij.legacy.isActive():
            imp = self._ij.WindowManager.getCurrentImage()
        if imp is None:
            return
        stack = imp.getStack()
        pixels = imp.getProcessor().getPixels()
        stack.setPixels(pixels, imp.getCurrentSlice())

    def to_dataset(self, data, dim_order=None):
        """Convert the data into an ImageJ2 Dataset.

        Converts a Python image (e.g. xarray or numpy array) or Java image (e.g.
        RandomAccessibleInterval or Img) into a net.imagej.Dataset Java object.

        :param data: Image object to be converted to Dataset.
        :return: A net.imagej.Dataset.
        """
        if sj.isjava(data):
            if dim_order:
                _logger.warning(
                    f"Dimension reordering is not supported for {type(data)}."
                )
            return convert.java_to_dataset(self._ij, data)

        if images.is_xarraylike(data):
            return convert.xarray_to_dataset(
                self._ij, convert._rename_xarray_dims(data, dim_order)
            )
        if images.is_arraylike(data):
            return convert.xarray_to_dataset(
                self._ij, convert.ndarray_to_xarray(data, dim_order)
            )

        raise TypeError(f"Type not supported: {type(data)}")

    def to_img(self, data, dim_order=None):
        """Convert the data into an ImgLib2 Img.

        Converts a Python image (e.g. xarray or numpy array) or Java image (e.g.
        RandomAccessibleInterval) into a net.imglib2.img.Img Java object.

        :param data: Image object to be converted to Img.
        :return: A net.imglib2.img.Img.
        """
        if sj.isjava(data):
            if dim_order:
                _logger.warning(
                    f"Dimension reordering is not supported for {type(data)}."
                )
            return convert.java_to_img(self._ij, data)

        if images.is_xarraylike(data):
            return convert.xarray_to_img(
                self._ij, convert._rename_xarray_dims(data, dim_order)
            )
        if images.is_arraylike(data):
            return convert.xarray_to_img(
                self._ij, convert.ndarray_to_xarray(data, dim_order)
            )

        raise TypeError(f"Type not supported: {type(data)}")

    def to_imageplus(self, data):
        """Convert the data into an ImageJ ImagePlus.

        Converts a Python image (e.g. xarray or numpy array) or Java image (e.g.
        RandomAccessibleInterval or Dataset) into an ij.ImagePlus Java object.

        :param data: Image object to be converted to ImagePlus.
        :return: An ij.ImagePlus.
        """
        self._ij._check_legacy_active("Conversion to ImagePlus is not supported.")
        return self._ij.convert().convert(self.to_dataset(data), jc.ImagePlus)

    def to_java(self, data, **hints):
        """Convert supported Python data into Java equivalents.

        Converts Python objects (e.g. xarray.DataArray) into the Java
        equivalents. For numpy arrays, the Java image points to the Python array.

        :param data: Python object to be converted into its respective Java counterpart.
        :param hints: Optional conversion hints.
        :return: A Java object converted from Python.
        """

        return sj.to_java(data, **hints)

    def to_xarray(self, data, dim_order=None):
        """Convert the data into an ImgLib2 Img.

        Converts a Python image (e.g. xarray or numpy array) or Java image (e.g.
        RandomAccessibleInterval) into an xarray.DataArray Python object.

        :param data: Image object to be converted to xarray.DataArray.
        :return: An xarray.DataArray.
        """
        if sj.isjava(data):
            if dim_order:
                _logger.warning(f"Conversion hints are not supported for {type(data)}.")
            if jc.ImagePlus and isinstance(data, jc.ImagePlus):
                data = convert.imageplus_to_imgplus(self._ij, data)
            if convert.supports_java_to_xarray(self._ij, data):
                return convert.java_to_xarray(self._ij, data)
            if convert.supports_java_to_ndarray(self._ij, data):
                return convert.ndarray_to_xarray(
                    convert.java_to_ndarray(self._ij, data)
                )

        if images.is_xarraylike(data):
            return convert._rename_xarray_dims(data, dim_order)
        if images.is_arraylike(data):
            return convert.ndarray_to_xarray(data, dim_order)

        raise TypeError(f"Type not supported: {type(data)}.")

    # -- Deprecated methods --

    def active_image_plus(self, sync=True) -> "jc.ImagePlus":
        """
        ij.py.active_image_plus() is deprecated.
        Use ij.py.active_imageplus() instead.
        """
        _logger.warning(
            "ij.py.active_image_plus() is deprecated. "
            "Use ij.py.active_imageplus() instead."
        )
        return self.active_imageplus(sync)

    def dims(self, image):
        """
        ij.py.dims(image) is deprecated. Use image.shape instead.
        """
        _logger.warning("ij.py.dims(image) is deprecated. Use image.shape instead.")
        if images.is_arraylike(image):
            return image.shape
        if not sj.isjava(image):
            raise TypeError("Unsupported type: " + str(type(image)))
        if isinstance(image, jc.Dimensions):
            return list(image.dimensionsAsLongArray())
        if jc.ImagePlus and isinstance(image, jc.ImagePlus):
            dims = image.getDimensions()
            dims.reverse()
            dims = [dim for dim in dims if dim > 1]
            return dims
        raise TypeError("Unsupported Java type: " + str(sj.jclass(image).getName()))

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
            "ij.py.new_numpy_image() is deprecated. "
            "Use ij.py.initialize_numpy_image() instead."
        )
        return np.zeros(self.dims(image), dtype=dtype_to_use)

    def synchronize_ij1_to_ij2(self, imp: "jc.ImagePlus"):
        """
        This function is deprecated. Use sync_image instead.
        """
        _logger.warning(
            "The synchronize_ij1_to_ij2 function is deprecated. Use sync_image instead."
        )
        self.sync_image(imp)

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

    def _add_converters(self):
        """Add all known converters to ScyJava's conversion mechanism."""

        # Python to Java
        sj.add_java_converter(
            sj.Converter(
                predicate=images.is_xarraylike,
                converter=lambda obj, **hints: self.to_dataset(obj),
                priority=sj.Priority.HIGH + 1,
            )
        )
        sj.add_java_converter(
            sj.Converter(
                predicate=convert.supports_ctype_to_realtype,
                converter=convert.ctype_to_realtype,
                priority=sj.Priority.HIGH + 1,
            )
        )
        sj.add_java_converter(
            sj.Converter(
                predicate=convert.supports_labeling_to_imglabeling,
                converter=lambda obj: convert.labeling_to_imglabeling(self._ij, obj),
                priority=sj.Priority.HIGH + 1,
            )
        )
        sj.add_java_converter(
            sj.Converter(
                predicate=images.is_memoryarraylike,
                converter=lambda obj, **hints: self.to_img(obj),
                priority=sj.Priority.HIGH,
            )
        )

        # Java to Python
        sj.add_py_converter(
            sj.Converter(
                predicate=lambda obj: jc.ImagePlus and isinstance(obj, jc.ImagePlus),
                converter=lambda obj: self.from_java(
                    convert.imageplus_to_imgplus(self._ij, obj)
                ),
                priority=sj.Priority.HIGH + 2,
            )
        )
        sj.add_py_converter(
            sj.Converter(
                predicate=convert.supports_realtype_to_ctype,
                converter=convert.realtype_to_ctype,
                priority=sj.Priority.HIGH + 1,
            )
        )
        sj.add_py_converter(
            sj.Converter(
                predicate=lambda obj: convert.supports_java_to_xarray(self._ij, obj),
                converter=lambda obj: convert.java_to_xarray(self._ij, obj),
                priority=sj.Priority.HIGH,
            )
        )
        sj.add_py_converter(
            sj.Converter(
                predicate=convert.supports_imglabeling_to_labeling,
                converter=lambda obj: convert.imglabeling_to_labeling(self._ij, obj),
                priority=sj.Priority.HIGH,
            )
        )
        sj.add_py_converter(
            sj.Converter(
                predicate=lambda obj: convert.supports_java_to_ndarray(self._ij, obj),
                converter=lambda obj: convert.java_to_ndarray(self._ij, obj),
                priority=sj.Priority.HIGH - 2,
            )
        )
        sj.add_py_converter(
            sj.Converter(
                predicate=lambda obj: isinstance(obj, jc.ImageMetadata),
                converter=lambda obj: convert.image_metadata_to_dict(self._ij, obj),
                priority=sj.Priority.HIGH - 2,
            )
        )
        sj.add_py_converter(
            sj.Converter(
                predicate=lambda obj: isinstance(obj, jc.MetadataWrapper),
                converter=lambda obj: convert.metadata_wrapper_to_dict(self._ij, obj),
                priority=sj.Priority.HIGH - 2,
            )
        )
        # add the ij.measure.ResultsTable converter only if legacy is enabled
        if self._ij.legacy and self._ij.legacy.isActive():
            sj.add_py_converter(
                sj.Converter(
                    predicate=lambda obj: isinstance(obj, jc.ResultsTable),
                    converter=lambda obj: self.from_java(
                        convert.results_table_to_scijava_table(self._ij, obj)
                    ),
                    priority=sj.Priority.HIGH + 2,
                )
            )

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
        if isinstance(value, jc.ImagePlus):
            return str(value.getTitle())
        temp_value = str(value).replace("\\", "/")
        if temp_value.startswith("[") and temp_value.endswith("]"):
            return temp_value
        final_value = "[" + temp_value + "]"
        return final_value

    def _get_origin(self, axis):
        """
        Get the coordinate origin of an axis, assuming it is the first entry.
        :param axis: A 1D list like entry accessible with indexing, which contains
            the axis coordinates
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
                # NB This call is necessary for loading the LegacyService
                sj.jimport("net.imagej.legacy.LegacyService")

                self._legacy = self.get("net.imagej.legacy.LegacyService")
                if self.ui().isHeadless():
                    _logger.warning(
                        "Operating in headless mode - the original ImageJ "
                        "will have limited functionality."
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
                    f"Operating in headless mode - the {class_name} "
                    "class will not be fully functional."
                )
            setattr(self, property_name, sj.jimport(fqcn))

        return getattr(self, property_name)

    def _check_legacy_active(self, usage_context=""):
        if not self.legacy or not self.legacy.isActive():
            raise ImportError(
                "The original ImageJ is not available in this environment. "
                f"{usage_context} See: "
                "https://github.com/imagej/pyimagej/blob/main/doc/Initialization.md"
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

    def _compute(self, other, ifunc):
        return jc.ImgMath.computeInto(ifunc(self._jargs(self, other)), self.copy())

    def __add__(self, other):
        """Return self + value."""
        return (
            self._op.run("math.add", self, other)
            if self._op is not None
            else self._compute(other, jc.ImgMath.add)
        )

    def __sub__(self, other):
        """Return self - value."""
        return (
            self._op.run("math.sub", self, other)
            if self._op is not None
            else self._compute(other, jc.ImgMath.sub)
        )

    def __mul__(self, other):
        """Return self * value."""
        return (
            self._op.run("math.mul", self, other)
            if self._op is not None
            else self._compute(other, jc.ImgMath.mul)
        )

    def __truediv__(self, other):
        """Return self / value."""
        return (
            self._op.run("math.div", self, other)
            if self._op is not None
            else self._compute(other, jc.ImgMath.div)
        )

    def __getitem__(self, key):
        if isinstance(key, slice):
            # Wrap single slice into tuple of length 1.
            return self._slice((key,))
        elif isinstance(key, tuple):
            return self._index(key) if self._is_index(key) else self._slice(key)
        elif isinstance(key, int):
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
        return type(jc.Util.getTypeFromInterval(self))

    def squeeze(self, axis=None):
        """Remove axes of length one from array.

        :return: Squeezed RandomAccessibleInterval.
        """
        if axis is None:
            # Process all dimensions.
            axis = tuple(range(self.numDimensions()))
        if isinstance(axis, int):
            # Convert int to singleton tuple.
            axis = (axis,)
        if not isinstance(axis, tuple):
            raise ValueError(f"Invalid type for axis parameter: {type(axis)}")

        res = self
        for d in range(self.numDimensions() - 1, -1, -1):
            if d in axis and self.dimension(d) == 1:
                res = jc.Views.hyperSlice(res, d, self.min(d))
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
                view = self._op.run("transform.permuteView", view, i, max_dim - i)
            else:
                view = jc.Views.permute(view, i, max_dim - i)
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

    def _is_index(self, a):
        # Check dimensionality - if we don't have enough dims, it's a slice
        num_dims = 1 if isinstance(a, int) else len(a)
        if num_dims < self.numDimensions():
            return False
        # if an int, it is an index
        if isinstance(a, int):
            return True
        # if we have a tuple, it's an index if there are any slices
        hasSlice = True in [isinstance(item, slice) for item in a]
        return not hasSlice

    def _jargs(self, *args):
        return JObjectArray()(list(map(sj.to_java, args)))

    @property
    @lru_cache(maxsize=None)
    def _op(self):
        return (
            self.getContext().getService("net.imagej.ops.OpService")
            if hasattr(self, "getContext")
            else None
        )

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
        dslices = [r if isinstance(r, slice) else slice(r, r + 1) for r in ranges]
        for dslice in dslices:
            imax.append(None if dslice.stop is None else dslice.stop - 1)
            imin.append(None if dslice.start is None else dslice.start)
            istep.append(1 if dslice.step is None else dslice.step)

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
    def dim_axes(self) -> Tuple["jc.Axis"]:
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
    mode: Union[Mode, str] = Mode.HEADLESS,
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
        |   [
        |       "net.imagej:imagej:2.3.0",
        |       "net.imagej:imagej-legacy",
        |       "net.preibisch:BigStitcher",
        |   ]
        | ).
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

        ij = imagej.init("sc.fiji:fiji", mode=imagej.Mode.GUI)
    """
    if headless is not None:
        _logger.warning(
            "The headless flag of imagej.init is deprecated. "
            "Use the mode argument instead."
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
                        advice += (
                            "E.g.: conda install -c conda-forge pyobjc-core "
                            "pyobjc-framework-cocoa"
                        )
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
    """
    Entry point for launching ImageJ from the command line via the `imagej`
    console entry point script.
    """
    args = []
    for i in range(1, len(sys.argv)):
        args.append(sys.argv[i])
    mode = "headless" if "--headless" in args else "gui"
    # Initialize imagej
    init(mode=mode)


def _create_gateway():
    # Initialize ImageJ2
    try:
        ImageJ = jc.ImageJ
    except TypeError:
        _logger.error(
            "***Invalid initialization: ImageJ2 was not found***\n"
            "Please update your initialization call to include an ImageJ2 "
            "application or endpoint (e.g. net.imagej:imagej).\n"
            "NOTE: You MUST restart your python interpreter as Java can only "
            "be started once."
        )
        return False

    ij = ImageJ()

    # Register a Python-side script runner object, used by the
    # org.scijava:scripting-python script language plugin.
    if callable(getattr(sj, "enable_python_scripting", None)):
        sj.enable_python_scripting(ij.context())

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
            sj.config.add_option("--add-opens=java.desktop/sun.awt.X11=ALL-UNNAMED")
    except RuntimeError as e:
        _logger.warning("Failed to guess the Java version.")
        _logger.debug(e, exc_info=True)

    # We want ImageJ2's endpoints to come first, so these will be restored
    # later
    original_endpoints = sj.config.endpoints.copy()
    sj.config.endpoints.clear()
    init_failed = False

    if ij_dir_or_version_or_endpoint is None:
        # Use latest release of ImageJ2.
        _logger.debug("Using newest ImageJ2 release")
        sj.config.endpoints.append("net.imagej:imagej")

    elif isinstance(ij_dir_or_version_or_endpoint, list):
        # Looks like a list of Maven endpoints.
        _logger.debug(
            "List of Maven coordinates given: %s", ij_dir_or_version_or_endpoint
        )
        if _includes_imagej_legacy(
            ij_dir_or_version_or_endpoint
        ) or _includes_imagej_legacy(original_endpoints):
            add_legacy = False
        sj.config.endpoints.extend(ij_dir_or_version_or_endpoint)

    elif os.path.isdir(os.path.expanduser(ij_dir_or_version_or_endpoint)):
        # Looks like a path to a local ImageJ2 installation.
        path = os.path.abspath(os.path.expanduser(ij_dir_or_version_or_endpoint))
        _logger.debug("Local path to ImageJ2 installation given: %s", path)
        add_legacy = False
        num_jars = _set_ij_env(path)
        if num_jars <= 0:
            _logger.error(
                "Given directory does not appear to be a valid ImageJ2 installation: "
                f"{path}"
            )
            init_failed = True
        else:
            _logger.info("Added " + str(num_jars + 1) + " JARs to the Java classpath.")
            plugins_dir = str(Path(path) / "plugins")
            sj.config.add_option("-Dplugins.dir=" + plugins_dir)

    elif re.match("^(/|[A-Za-z]:)", ij_dir_or_version_or_endpoint):
        # Looks like a file path was intended, but it's not a folder.
        path = ij_dir_or_version_or_endpoint
        _logger.error("Local path given is not a directory: %s", path)
        init_failed = True

    elif ":" in ij_dir_or_version_or_endpoint:
        # Looks like an artifact endpoint.
        _logger.debug("Maven coordinate given: %s", ij_dir_or_version_or_endpoint)
        # Strip whitespace and split concatenated endpoints.
        endpoints = re.sub("\\s*", "", ij_dir_or_version_or_endpoint).split("+")
        if _includes_imagej_legacy(endpoints) or _includes_imagej_legacy(
            original_endpoints
        ):
            add_legacy = False
        sj.config.endpoints.extend(endpoints)

    elif re.match("\\d+\\.\\d+\\.\\d+", ij_dir_or_version_or_endpoint):
        # Looks like an x.y.z-style version of net.imagej:imagej.
        _logger.debug("ImageJ2 version given: %s", ij_dir_or_version_or_endpoint)
        sj.config.endpoints.append("net.imagej:imagej:" + ij_dir_or_version_or_endpoint)

    else:
        # String is in an unknown form.
        _logger.error(
            "Invalid initialization string: %s", ij_dir_or_version_or_endpoint
        )
        init_failed = True

    if init_failed:
        # Restore any pre-existing endpoints to allow for re-initialization.
        sj.config.endpoints.clear()
        sj.config.endpoints.extend(original_endpoints)
        return False

    if len(sj.config.endpoints) > 0:
        # NB: Only append MANAGED components when there is at least one
        # component already on the list. Otherwise, there can be nothing with
        # the needed dependencyManagement section. In particular, this
        # situation arises when wrapping a local ImageJ2 installation.

        if add_legacy:
            sj.config.endpoints.append("net.imagej:imagej-legacy:MANAGED")

        # Add SciJava logging configuration. Without this, we see the warning:
        # SLF4J: Failed to load class "org.slf4j.impl.StaticLoggerBinder".
        # SLF4J: Defaulting to no-operation (NOP) logger implementation
        sj.config.endpoints.append("org.scijava:scijava-config:MANAGED")

        # Add additional ImageJ endpoints specific to PyImageJ.

        # Try to glean the needed ImageJ2 version.
        for coord in sj.config.endpoints:
            if not re.match("(net.imagej:imagej|sc.fiji:fiji)(:|$)", coord):
                # not an ImageJ2 or Fiji coordinate.
                continue

            # Extract major and minor version digits.
            gav = coord.split(":")
            version = gav[2] if len(gav) > 2 else "RELEASE"
            digits = version.split(".") if version else []
            if len(digits) == 0:
                # Unparseable version; skip this coordinate.
                continue
            minor_digit = None
            try:
                minor_digit = int(digits[1]) if len(digits) > 1 else -1
            except ValueError:
                pass

            # Now some case logic to figure which scifio-labeling version to use.
            scifio_labeling_version = None
            if version == "RELEASE":
                # Using newest ImageJ2, therefore also use newest scifio-labeling.
                scifio_labeling_version = "RELEASE"
            elif version.startswith("2."):
                # For v2.10.0+, scifio-labeling is managed; before that, must hardcode.
                scifio_labeling_version = "MANAGED" if minor_digit >= 10 else "0.3.1"

            # Add scifio-labeling if appropriate.
            if scifio_labeling_version:
                sj.config.endpoints.append(
                    f"io.scif:scifio-labeling:{scifio_labeling_version}"
                )

    # Restore any pre-existing endpoints, after ImageJ2's.
    sj.config.endpoints.extend(original_endpoints)

    try:
        sj.start_jvm()
    except subprocess.CalledProcessError as e:
        # Check to see if initialization failed due to "un-managed"
        # imagej-legacy
        err_lines = []
        unmanaged_legacy = False
        if e.stdout:
            err_lines += e.stdout.decode().splitlines()
        if e.stderr:
            err_lines += e.stderr.decode().splitlines()
        for line in err_lines:
            if (
                "'dependencies.dependency.version' for net.imagej:imagej-legacy:jar "
                "is missing." in line
            ):
                unmanaged_legacy = True
        if unmanaged_legacy:
            _logger.error(
                "***Invalid Initialization: you may be using a primary endpoint that "
                "lacks pom-scijava as a parent***\n"
                "To keep all Java components at compatible versions we recommend using "
                "a primary endpoint with a pom-scijava parent.\n"
                "For example, by putting 'net.imagej:imagej' first in your list of "
                "endpoints.\n"
                "If you are sure you DO NOT want a primary endpoint with a pom-scijava "
                "parent, please re-initialize with 'add_legacy=False'."
            )
        sj.config.endpoints.clear()
        sj.config.endpoints.extend(original_endpoints)
        return False

    return True


def _includes_imagej_legacy(items: list):
    return any(item.startswith("net.imagej:imagej-legacy") for item in items)


def _set_ij_env(ij_dir):
    """
    Create a list of required jars and add to the java classpath.

    :param ij_dir: System path for Fiji.app.
    :return: num_jar(int): Number of jars added.
    """
    jars = []
    # search jars directory
    jars.extend(find_jars(ij_dir + "/jars"))
    # search plugins directory
    jars.extend(find_jars(ij_dir + "/plugins"))
    # add to classpath
    sj.config.add_classpath(os.pathsep.join(jars))
    return len(jars)
