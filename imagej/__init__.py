"""
PyImageJ provides a set of wrapper functions for integration between ImageJ
and Python. A major advantage of this approach is the ability to combine
ImageJ with other tools available from the Python software ecosystem,
including NumPy, SciPy, scikit-image, CellProfiler, OpenCV, ITK and more.

The first step when using pyimagej is to create an ImageJ gateway.
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

import logging
import os
import re
import sys

import imglyb
import numpy as np
import scyjava as sj
import xarray as xr

from pathlib import Path

from jpype import JArray, JException, JImplementationFor, JObject

from .config import __author__, __version__

_logger = logging.getLogger(__name__)

# Enable debug logging if DEBUG environment variable is set.
try:
    debug = os.environ['DEBUG']
    if debug:
        _logger.setLevel(logging.DEBUG)
except KeyError as e:
    pass


def _dump_exception(exc):
    if _logger.isEnabledFor(logging.DEBUG):
        jtrace = jstacktrace(exc)
        if jtrace:
            _logger.debug(jtrace)


def _search_for_jars(target_dir, subfolder=''):
    """
    Search and recursively add .jar files to a list.

    :param target_dir: Base path to search.
    :param subfolder: Optional sub-directory to start the search.
    :return: A list of jar files.
    """
    jars = []
    for root, dirs, files in os.walk(target_dir + subfolder):
        for f in files:
            if f.endswith('.jar'):
                path = root + '/' + f
                jars.append(path)
                _logger.debug('Added %s', path)
    return jars


def _set_ij_env(ij_dir):
    """
    Create a list of required jars and add to the java classpath.

    :param ij_dir: System path for Fiji.app.
    :return: num_jar(int): Number of jars added.
    """
    jars = []
    # search jars directory
    jars.extend(_search_for_jars(ij_dir, '/jars'))
    # search plugins directory
    jars.extend(_search_for_jars(ij_dir, '/plugins'))
    # add to classpath
    sj.config.add_classpath(os.pathsep.join(jars))
    return len(jars)

def init(ij_dir_or_version_or_endpoint=None, headless=True):
    """Initialize the ImageJ environment.

    Initialize the ImageJ enviornment with a local ImageJ installation,
    a specific version of ImageJ, or with maven artifacts. ImageJ can
    be initialized in headless mode or GUI mode (note: not all ImageJ
    operations function in headless mode).

    :param ij_dir_or_version_or_endpoint:
        Path to a local ImageJ installation (e.g. /Applications/Fiji.app),
        OR version of net.imagej:imagej artifact to launch (e.g. 2.0.0-rc-67),
        OR endpoint of another artifact (e.g. sc.fiji:fiji) that uses imagej.
        OR list of Maven artifacts to include (e.g. ['net.imagej:imagej-legacy', 'net.preibisch:BigStitcher'])
    :param headless: Whether to start the JVM in headless or gui mode.
    :return: An instance of the net.imagej.ImageJ gateway

    :example:

    ij = imagej.init('sc.fiji:fiji', headless=False)
    """

    global ij

    # Check if JPype JVM is already running
    if sj.jvm_started():
        _logger.debug('The JPype JVM is already running.')
    else:
        # Initialize configuration.
        if headless:
            sj.config.add_option('-Djava.awt.headless=true')

        # We want ImageJ's endpoints to come first, so these will be restored later
        original_endpoints = sj.config.endpoints.copy()
        sj.config.endpoints.clear()

        if ij_dir_or_version_or_endpoint is None:
            # Use latest release of ImageJ.
            _logger.debug('Using newest ImageJ release')
            sj.config.endpoints.extend(('net.imagej:imagej', 'net.imagej:imagej-legacy'))

        elif isinstance(ij_dir_or_version_or_endpoint, list):
            # Assume that this is a list of Maven endpoints
            endpoint = '+'.join(ij_dir_or_version_or_endpoint)
            _logger.debug('List of Maven coordinates given: %s', ij_dir_or_version_or_endpoint)
            sj.config.endpoints.append(endpoint)

        elif os.path.isdir(ij_dir_or_version_or_endpoint):
            # Assume path to local ImageJ installation.
            path = ij_dir_or_version_or_endpoint
            _logger.debug('Local path to ImageJ installation given: %s', path)
            num_jars = _set_ij_env(path)
            if num_jars <= 0:
                _logger.error('Given directory does not appear to be a valid ImageJ installation: %s', path)
                return False
            _logger.info("Added " + str(num_jars + 1) + " JARs to the Java classpath.")
            plugins_dir = str(Path(path, 'plugins'))
            jvm_options = '-Dplugins.dir=' + plugins_dir

        elif re.match('^(/|[A-Za-z]:)', ij_dir_or_version_or_endpoint):
            # Looks like a file path was intended, but it's not a folder.
            path = ij_dir_or_version_or_endpoint
            _logger.error('Local path given is not a directory: %s', path)
            return False

        elif ':' in ij_dir_or_version_or_endpoint:
            # Assume endpoint of an artifact.
            # Strip out white spaces
            endpoint = ij_dir_or_version_or_endpoint.replace("    ", "")
            _logger.debug('Maven coordinate given: %s', endpoint)
            sj.config.endpoints.extend((endpoint, 'net.imagej:imagej-legacy'))

        else:
            # Assume version of net.imagej:imagej.
            version = ij_dir_or_version_or_endpoint
            # Skip ignore
            if not re.match('\d+\.\d+\.\d+', version):
                _logger.error('Invalid initialization string: %s', version)
                return False
            _logger.debug('ImageJ version given: %s', version)
            sj.config.endpoints.append('net.imagej:imagej:' + version)

        # Restore any pre-existing endpoints, after ImageJ's
        sj.config.endpoints.extend(original_endpoints)

    sj.start_jvm()

    JObjectArray = JArray(JObject)

    # Initialize ImageJ
    ImageJ = sj.jimport('net.imagej.ImageJ')
    ij = ImageJ()

    # Append some useful utility functions to the ImageJ gateway.
    Dataset                  = sj.jimport('net.imagej.Dataset')
    ImgPlus                  = sj.jimport('net.imagej.ImgPlus')
    Img                      = sj.jimport('net.imglib2.img.Img')
    RandomAccessibleInterval = sj.jimport('net.imglib2.RandomAccessibleInterval')
    Axes                     = sj.jimport('net.imagej.axis.Axes')
    Double                   = sj.jimport('java.lang.Double')

    # EnumeratedAxis is a new axis made for xarray, so is only present in
    # ImageJ versions that are released later than March 2020. This check
    # defaults to LinearAxis instead if Enumerated does not work.
    try:
        EnumeratedAxis           = sj.jimport('net.imagej.axis.EnumeratedAxis')
    except (JException, TypeError):
        DefaultLinearAxis = sj.jimport('net.imagej.axis.DefaultLinearAxis')
        def EnumeratedAxis(axis_type, values):
            origin = values[0]
            scale = values[1] - values[0]
            axis = DefaultLinearAxis(axis_type, scale, origin)
            return axis

    class ImageJPython:
        def __init__(self, ij):
            self._ij = ij

        def dims(self, image):
            """Return the dimensions of the input image.

            Return the dimenions (i.e. shape) of an input numpy array,
            ImgLib2 image or an ImageJ ImagePlus.

            :param image:
                A numpy array.
                OR An ImgLib2 image ('net.imglib2.Interval').
                OR An ImageJ2 Dataset ('net.imagej.Dataset').
                OR An ImageJ ImagePlus ('ij.ImagePlus').
            :param return: Dimensions of the input image.
            """
            if self._is_arraylike(image):
                return image.shape
            if not sj.isjava(image):
                raise TypeError('Unsupported type: ' + str(type(image)))
            if sj.jclass('net.imglib2.Dimensions').isInstance(image):
                return [image.dimension(d) for d in range(image.numDimensions() -1, -1, -1)]
            if sj.jclass('ij.ImagePlus').isInstance(image):
                dims = image.getDimensions()
                dims.reverse()
                dims = [dim for dim in dims if dim > 1]
                return dims
            raise TypeError('Unsupported Java type: ' + str(sj.jclass(image).getName()))

        def dtype(self, image_or_type):
            """Return the dtype of the input image.

            Return the dtype of an input numpy array, ImgLib2 image
            or an Image ImagePlus.

            :param image_or_type:
                A numpy array.
                OR A numpy array dtype.
                OR An ImgLib2 image ('net.imglib2.Interval').
                OR An ImageJ2 Dataset ('net.imagej.Dataset').
                OR An ImageJ ImagePlus ('ij.ImagePlus').
            :param return: Input image dtype.
            """
            if type(image_or_type) == np.dtype:
                return image_or_type
            if self._is_arraylike(image_or_type):
                return image_or_type.dtype
            if not sj.isjava(image_or_type):
                raise TypeError('Unsupported type: ' + str(type(image_or_type)))

            # -- ImgLib2 types --
            if sj.jclass('net.imglib2.type.Type').isInstance(image_or_type):
                ij2_types = {
                    #'net.imglib2.type.logic.BitType':                                'bool',
                    'net.imglib2.type.numeric.integer.ByteType':                     'int8',
                    'net.imglib2.type.numeric.integer.ByteLongAccessType':           'int8',
                    'net.imglib2.type.numeric.integer.ShortType':                    'int16',
                    'net.imglib2.type.numeric.integer.ShortLongAccessType':          'int16',
                    'net.imglib2.type.numeric.integer.IntType':                      'int32',
                    'net.imglib2.type.numeric.integer.IntLongAccessType':            'int32',
                    'net.imglib2.type.numeric.integer.LongType':                     'int64',
                    'net.imglib2.type.numeric.integer.LongLongAccessType':           'int64',
                    'net.imglib2.type.numeric.integer.UnsignedByteType':             'uint8',
                    'net.imglib2.type.numeric.integer.UnsignedByteLongAccessType':   'uint8',
                    'net.imglib2.type.numeric.integer.UnsignedShortType':            'uint16',
                    'net.imglib2.type.numeric.integer.UnsignedShortLongAccessType':  'uint16',
                    'net.imglib2.type.numeric.integer.UnsignedIntType':              'uint32',
                    'net.imglib2.type.numeric.integer.UnsignedIntLongAccessType':    'uint32',
                    'net.imglib2.type.numeric.integer.UnsignedLongType':             'uint64',
                    'net.imglib2.type.numeric.integer.UnsignedLongLongAccessType':   'uint64',
                    #'net.imglib2.type.numeric.ARGBType':                             'argb',
                    #'net.imglib2.type.numeric.ARGBLongAccessType':                   'argb',
                    'net.imglib2.type.numeric.real.FloatType':                       'float32',
                    'net.imglib2.type.numeric.real.FloatLongAccessType':             'float32',
                    'net.imglib2.type.numeric.real.DoubleType':                      'float64',
                    'net.imglib2.type.numeric.real.DoubleLongAccessType':            'float64',
                    #'net.imglib2.type.numeric.complex.ComplexFloatType':             'cfloat32',
                    #'net.imglib2.type.numeric.complex.ComplexFloatLongAccessType':   'cfloat32',
                    #'net.imglib2.type.numeric.complex.ComplexDoubleType':            'cfloat64',
                    #'net.imglib2.type.numeric.complex.ComplexDoubleLongAccessType':  'cfloat64',
                }
                for c in ij2_types:
                    if sj.jclass(c).isInstance(image_or_type):
                        return np.dtype(ij2_types[c])
                raise TypeError('Unsupported ImgLib2 type: {}'.format(image_or_type))

            # -- ImgLib2 images --
            if sj.jclass('net.imglib2.IterableInterval').isInstance(image_or_type):
                ij2_type = image_or_type.firstElement()
                return self.dtype(ij2_type)
            if sj.jclass('net.imglib2.RandomAccessibleInterval').isInstance(image_or_type):
                Util = sj.jimport('net.imglib2.util.Util')
                ij2_type = Util.getTypeFromInterval(image_or_type)
                return self.dtype(ij2_type)

            # -- ImageJ1 images --
            if sj.jclass('ij.ImagePlus').isInstance(image_or_type):
                ij1_type = image_or_type.getType()
                ImagePlus = sj.jimport('ij.ImagePlus')
                ij1_types = {
                    ImagePlus.GRAY8:  'uint8',
                    ImagePlus.GRAY16: 'uint16',
                    ImagePlus.GRAY32: 'float32', # NB: ImageJ1's 32-bit type is float32, not uint32.
                }
                for t in ij1_types:
                    if ij1_type == t:
                        return np.dtype(ij1_types[t])
                raise TypeError('Unsupported ImageJ1 type: {}'.format(ij1_type))

            raise TypeError('Unsupported Java type: ' + str(sj.jclass(image_or_type).getName()))

        def new_numpy_image(self, image):
            """Create an empty numpy array.

            Create a new numpy array with the same shape and
            type as the input image, filled with zeros.

            :param image: A numpy array.
            :return: A new zero filled array of given shape and type.
            """
            try:
                dtype_to_use = self.dtype(image)
            except TypeError:
                dtype_to_use = np.dtype('float64')
            return np.zeros(self.dims(image), dtype=dtype_to_use)

        def rai_to_numpy(self, rai):
            """Convert a RandomAccessibleInterval into a numpy array.

            Convert a RandomAccessibleInterval ('net.imglib2.RandomAccessibleInterval') 
            into a new numpy array.

            :param rai: A RandomAccisbleInterval ('net.imglib2.RandomAccessibleInterval').
            :return: A numpy array of the input RandomAccisbleInterval.
            """
            # check imagej-common version for fast copy availability.
            ijc_slow_copy_version = '0.30.0'
            ijc_active_version = sj.get_version(Dataset)
            fast_copy_available = sj.compare_version(ijc_slow_copy_version, ijc_active_version)

            if fast_copy_available:
                Images = sj.jimport('net.imagej.util.Images')
                result = self.new_numpy_image(rai)
                Images.copy(rai, self.to_java(result))
            else:
                result = self.new_numpy_image(rai)
                self._ij.op().run("copy.rai", self.to_java(result), rai)

            return result

        def run_plugin(self, plugin, args=None, ij1_style=True):
            """Run an ImageJ plugin.

            Run an ImageJ plugin by specifiying the plugin name as a string,
            and the plugin arguments as a dictionary. For the few plugins that
            use the ImageJ2 style macros (i.e. explicit booleans in the recorder),
            set the option variable ij1_style=False.

            :param plugin: The string name for the plugin command.
            :param args: A dictionary of plugin arguments in key: value pairs.
            :param ij1_style: Boolean to set which implicit boolean style to use (ImageJ or ImageJ2).
            :return: Runs the specified plugin with the given arguments.

            :example:
            
            plugin = 'Mean'
            args = {
                'block_radius_x': 10,
                'block_radius_y': 10
            }
            ij.py.run_plugin(plugin, args)
            """
            macro = self._assemble_plugin_macro(plugin, args=args, ij1_style=ij1_style)
            return self.run_macro(macro)

        def run_macro(self, macro, args=None):
            """Run an ImageJ macro.

            Run an ImageJ macro by providing the macro code/script in a string and
            the arguments in a dictionary.

            :param macro: The macro code/script as a string.
            :param args: A dictionary of macro arguments in key: valye pairs.
            :return: Runs the specified macro with the given arguments.

            :example:

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
            if not ij.legacy.isActive():
                raise ImportError("Your IJ endpoint does not support IJ1, and thus cannot use IJ1 macros.")

            try:
                if args is None:
                    return self._ij.script().run("macro.ijm", macro, True).get()
                else:
                    return self._ij.script().run("macro.ijm", macro, True, ij.py.jargs(args)).get()
            except Exception as exc:
                _dump_exception(exc)
                raise exc

        def run_script(self, language, script, args=None):
            """Run an ImageJ script.

            Run a script in one of ImageJ's supported scripting languages.
            Specify the language of the script, provide the code as a string
            and the arguments as a dictionary.

            :param language: The file extension for the scripting language.
            :param script: A string of the script code.
            :param args: A dictionary of macro arguments in key: value pairs.
            :return: Runs the specified script with the given arguments.

            :example:

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
                raise ValueError("Script language '" + script_lang.getLanguageName() + "' has no extensions")
            ext = str(exts.get(0))
            try:
                if args is None:
                    return self._ij.script().run("script." + ext, script, True).get()
                return self._ij.script().run("script." + ext, script, True, ij.py.jargs(args)).get()
            except Exception as exc:
                _dump_exception(exc)
                raise exc

        def to_java(self, data):
            """Convert supported Python data into Java equivalents.

            Converts Python objects (e.g. xarray.DataArray) into the Java
            equivalents. For numpy arrays, the Java image points to the Python array.

            :param data: Python object to be converted into its respective Java counterpart.
            :return: A Java object convrted from Python.
            """
            if self._is_memoryarraylike(data):
                return imglyb.to_imglib(data)
            if self._is_xarraylike(data):
                return self.to_dataset(data)
            return sj.to_java(data)

        def to_dataset(self, data):
            """Converts the data into an ImageJ dataset"""
            if self._is_xarraylike(data):
                return self._xarray_to_dataset(data)
            if self._is_arraylike(data):
                return self._numpy_to_dataset(data)
            if sj.isjava(data):
                return self._java_to_dataset(data)

            raise TypeError(f'Type not supported: {type(data)}')

        def jargs(self, *args):
            """
            Converts Python arguments into a Java Object[] (i.e.: array of Java
            objects). This is particularly useful in combination with ImageJ's
            various run functions, including ij.command().run(...),
            ij.module().run(...), ij.script().run(...), and ij.op().run(...).
            :param args: The Python arguments to wrap into an Object[].
            """
            return JObjectArray([self.to_java(arg) for arg in args])

        def _numpy_to_dataset(self, data):
            rai = imglyb.to_imglib(data)
            return self._java_to_dataset(rai)

        def _ends_with_channel_axis(self, xarr):
            ends_with_axis = xarr.dims[len(xarr.dims)-1].lower() in ['c', 'channel']
            return ends_with_axis

        def _xarray_to_dataset(self, xarr):
            """
            Converts a xarray dataarray to a dataset, inverting C-style (slow axis first) to F-style (slow-axis last)
            :param xarr: Pass an xarray dataarray and turn into a dataset.
            :return: The dataset
            """
            if self._ends_with_channel_axis(xarr):
                vals = np.moveaxis(xarr.values, -1, 0)
                dataset = self._numpy_to_dataset(vals) # EE: investigate here....
            else:
                dataset = self._numpy_to_dataset(xarr.values)
            axes = self._assign_axes(xarr)
            dataset.setAxes(axes)
            self._assign_dataset_metadata(dataset, xarr.attrs)

            return dataset

        def _assign_axes(self, xarr):
            """
            Obtain xarray axes names, origin, and scale and convert into ImageJ Axis; currently supports EnumeratedAxis
            :param xarr: xarray that holds the units
            :return: A list of ImageJ Axis with the specified origin and scale
            """
            axes = ['']*len(xarr.dims)

            for axis in xarr.dims:
                axis_str = self._pydim_to_ijdim(axis)

                ax_type = Axes.get(axis_str)
                ax_num = self._get_axis_num(xarr, axis)

                scale = self._get_scale(xarr.coords[axis])
                if scale is None:
                    logging.warning(f"The {ax_type.label} axis is non-numeric and is translated to a linear index.")
                    doub_coords = [Double(np.double(x)) for x in np.arange(len(xarr.coords[axis]))]
                else:
                    doub_coords = [Double(np.double(x)) for x in xarr.coords[axis]]

                # EnumeratedAxis is a new axis made for xarray, so is only present in ImageJ versions that are released
                # later than March 2020.  This actually returns a LinearAxis if using an earlier version.
                java_axis = EnumeratedAxis(ax_type, ij.py.to_java(doub_coords))

                axes[ax_num] = java_axis

            return axes

        def _pydim_to_ijdim(self, axis):
            """Convert between the lowercase Python convention (x, y, z, c, t) to IJ (X, Y, Z, C, T)"""
            if str(axis) in ['x', 'y', 'z', 'c', 't']:
                return str(axis).upper()
            return str(axis)

        def _ijdim_to_pydim(self, axis):
            """Convert the IJ uppercase dimension convention (X, Y, Z C, T) to lowercase python (x, y, z, c, t) """
            if str(axis) in ['X', 'Y', 'Z', 'C', 'T']:
                return str(axis).lower()
            return str(axis)

        def _get_axis_num(self, xarr, axis):
            """
            Get the xarray -> java axis number due to inverted axis order for C style numpy arrays (default)
            :param xarr: Xarray to convert
            :param axis: Axis number to convert
            :return: Axis idx in java
            """
            py_axnum = xarr.get_axis_num(axis)
            if np.isfortran(xarr.values):
                return py_axnum

            if self._ends_with_channel_axis(xarr):
                if axis == len(xarr.dims) - 1:
                    return axis
                else:
                    return len(xarr.dims) - py_axnum - 2
            else:
                return len(xarr.dims) - py_axnum - 1

        def _assign_dataset_metadata(self, dataset, attrs):
            """
            :param dataset: ImageJ Java dataset
            :param attrs: Dictionary containing metadata
            """
            dataset.getProperties().putAll(self.to_java(attrs))

        def _get_origin(self, axis):
            """
            Get the coordinate origin of an axis, assuming it is the first entry.
            :param axis: A 1D list like entry accessible with indexing, which contains the axis coordinates
            :return: The origin for this axis.
            """
            return axis.values[0]

        def _get_scale(self, axis):
            """
            Get the scale of an axis, assuming it is linear and so the scale is simply second - first coordinate.
            :param axis: A 1D list like entry accessible with indexing, which contains the axis coordinates
            :return: The scale for this axis or None if it is a non-numeric scale.
            """
            try:
                return axis.values[1] - axis.values[0]
            except TypeError:
                return None

        def _java_to_dataset(self, data):
            """
            Converts the data into a ImageJ Dataset
            """
            # This try checking is necessary because the set of ImageJ converters is not complete.  E.g., here is no way
            # to directly go from Img to Dataset, instead you need to chain the Img->ImgPlus->Dataset converters.
            try:
                if self._ij.convert().supports(data, Dataset):
                    return self._ij.convert().convert(data, Dataset)
                if self._ij.convert().supports(data, ImgPlus):
                    imgPlus = self._ij.convert().convert(data, ImgPlus)
                    return self._ij.dataset().create(imgPlus)
                if self._ij.convert().supports(data, Img):
                    img = self._ij.convert().convert(data, Img)
                    return self._ij.dataset().create(ImgPlus(img))
                if self._ij.convert().supports(data, RandomAccessibleInterval):
                    rai = self._ij.convert().convert(data, RandomAccessibleInterval)
                    x = self._ij.dataset().create(rai)
                    return self._ij.dataset().create(rai)
            except Exception as exc:
                _dump_exception(exc)
                raise exc
            raise TypeError('Cannot convert to dataset: ' + str(type(data)))

        def from_java(self, data):
            """Convert supported Java data into Python equivalents.

            Converts Java objects (e.g. 'net.imagej.Dataset') into the Python
            equivalents.

            :param data: Java object to be converted into its respective Python counterpart.
            :return: A Python object convrted from Java.
            """
            # todo: convert a dataset to xarray

            if not sj.isjava(data): return data
            try:
                if self._ij.convert().supports(data, Dataset):
                    # HACK: Converter exists for ImagePlus -> Dataset, but not ImagePlus -> RAI.
                    data = self._ij.convert().convert(data, Dataset)
                    return self._dataset_to_xarray(data)
                if self._ij.convert().supports(data, RandomAccessibleInterval):
                    rai = self._ij.convert().convert(data, RandomAccessibleInterval)
                    return self.rai_to_numpy(rai)
            except Exception as exc:
                _dump_exception(exc)
                raise exc
            return sj.to_python(data)

        def _dataset_to_xarray(self, dataset):
            """
            Converts an ImageJ dataset into an xarray, inverting F-style (slow idx last) to C-style (slow idx first)
            :param dataset: ImageJ dataset
            :return: xarray with reversed (C-style) dims and coords as labeled by the dataset
            """
            attrs = self.from_java(dataset.getProperties())
            axes = [(JObject(dataset.axis(idx), sj.jimport('net.imagej.axis.CalibratedAxis')))
                    for idx in range(dataset.numDimensions())]

            dims = [self._ijdim_to_pydim(axes[idx].type().getLabel()) for idx in range(len(axes))]
            values = self.rai_to_numpy(dataset)
            coords = self._get_axes_coords(axes, dims, np.shape(np.transpose(values)))

            if dims[len(dims)-1].lower() in ['c', 'channel']:
                xarr_dims = self._invert_except_last_element(dims)
                values = np.moveaxis(values, 0, -1)
            else:
                xarr_dims = list(reversed(dims))

            xarr = xr.DataArray(values, dims=xarr_dims, coords=coords, attrs=attrs)
            return xarr

        def _invert_except_last_element(self, lst):
            """
            Invert a list except for the last element.
            :param lst:
            :return:
            """
            cut_list = lst[0:-1]
            reverse_cut = list(reversed(cut_list))
            reverse_cut.append(lst[-1])
            return reverse_cut

        def _get_axes_coords(self, axes, dims, shape):
            """
            Get xarray style coordinate list dictionary from a dataset
            :param axes: List of ImageJ axes
            :param dims: List of axes labels for each dataset axis
            :param shape: F-style, or reversed C-style, shape of axes numpy array.
            :return: Dictionary of coordinates for each axis.
            """
            coords = {dims[idx]: [axes[idx].calibratedValue(position) for position in range(shape[idx])]
                      for idx in range(len(dims))}
            return coords


        def show(self, image, cmap=None):
            """Display a Java or Python 2D image.
            
            Display a java or python 2D image.
            :param image: A Java or Python image that can be converted to a numpy array.
            :param cmap: The colormap of the image.
            :return: Displayed image.
            """
            if image is None:
                raise TypeError('Image must not be None')

            # NB: Import this only here on demand, rather than above.
            # Otherwise, some headless systems may experience errors
            # like "ImportError: Failed to import any qt binding".
            from matplotlib import pyplot

            pyplot.imshow(self.from_java(image), interpolation='nearest', cmap=cmap)
            pyplot.show()

        def _is_arraylike(self, arr):
            return hasattr(arr, 'shape') and \
                hasattr(arr, 'dtype') and \
                hasattr(arr, '__array__') and \
                hasattr(arr, 'ndim')

        def _is_memoryarraylike(self, arr):
            return self._is_arraylike(arr) and \
                hasattr(arr, 'data') and \
                type(arr.data).__name__ == 'memoryview'

        def _is_xarraylike(self, xarr):
            return hasattr(xarr, 'values') and \
                hasattr(xarr, 'dims') and \
                hasattr(xarr, 'coords') and \
                self._is_arraylike(xarr.values)

        def _assemble_plugin_macro(self, plugin: str, args=None, ij1_style=True):
            """
            Assemble an ImageJ macro string given a plugin to run and optional arguments in a dict
            :param plugin: The string call for the function to run
            :param args: A dict of macro arguments in key/value pairs
            :param ij1_style: Whether to use implicit booleans in IJ1 style or explicit booleans in IJ2 style
            :return: A string version of the macro run
            """
            if args is None:
                macro = "run(\"{}\");".format(plugin)
                return macro
            macro = """run("{0}", \"""".format(plugin)
            for key, value in args.items():
                argument = self._format_argument(key, value, ij1_style)
                if argument is not None:
                    macro = macro + ' {}'.format(argument)
            macro = macro + """\");"""
            return macro

        def _format_argument(self, key, value, ij1_style):
            if value is True:
                argument = '{}'.format(key)
                if not ij1_style:
                    argument = argument + '=true'
            elif value is False:
                argument = None
                if not ij1_style:
                    argument = '{0}=false'.format(key)
            elif value is None:
                raise NotImplementedError('Conversion for None is not yet implemented')
            else:
                val_str = self._format_value(value)
                argument = '{0}={1}'.format(key, val_str)
            return argument

        def _format_value(self, value):
            if sj.jclass('ij.ImagePlus').isInstance(value):
                return str(value.getTitle())
            temp_value = str(value).replace('\\', '/')
            if temp_value.startswith('[') and temp_value.endswith(']'):
                    return temp_value
            final_value = '[' + temp_value + ']'
            return final_value

        def window_manager(self):
            """
            ij.py.window_manager() is deprecated.
            Use ij.WindowManager instead.
            """
            logging.warning("ij.py.window_manager() is deprecated. Use ij.WindowManager instead.")
            return self._ij.WindowManager

        def active_xarray(self, sync=True):
            """Get the active image as an xarray.

            Convert the active image to a xarray.DataArray, synchronizing from ImageJ to ImageJ2.

            :param sync: Manually synchronize the current IJ1 slice if True.
            :return: numpy array containing the image data.
            """
            # todo: make the behavior use pure IJ2 if legacy is not active

            if ij.legacy.isActive():
                imp = self.active_image_plus(sync=sync)
                return self.from_java(imp)
            else:
                dataset = self.active_dataset()
                return self.from_java(dataset)

        def active_dataset(self):
            """Get the active Dataset image.

            Get the currently active Dataset from the Dataset service.
            :return: The Dataset corresponding to the active image.
            """
            return self._ij.imageDisplay().getActiveDataset()

        def active_image_plus(self, sync=True):
            """Get the active ImagePlus image.

            Get the currently active ImagePlus image, optionally synchronizing from ImageJ to ImageJ2.

            :param sync: Manually synchronize the current ImageJ slice if True.
            :return: The ImagePlus corresponding to the active image.
            """
            imp = self._ij.WindowManager.getCurrentImage()
            if imp is None: return None
            if sync:
                self.synchronize_ij1_to_ij2(imp)
            return imp

        def synchronize_ij1_to_ij2(self, imp):
            """ Syncronize data between ImageJ and ImageJ2.

            Synchronize between a Dataset or ImageDisplay linked to an 
            ImagePlus by accepting the ImagePlus data as true.

            :param imp: The IJ1 ImagePlus that needs to be synchronized.
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
            # Don't sync if the ImagePlus is not linked back to a corresponding dataset
            if str(type(pixels)) == '<class \'jnius.ByteArray\'>':
                return

            stack.setPixels(pixels, imp.getCurrentSlice())

    # attach ImageJPython to imagej
    imagejPythonObj = ImageJPython(ij)

    # Try to define the legacy service, and create a dummy method if it doesn't exist.

    # create the legacy service object

        #TODO here

    @JImplementationFor('net.imagej.ImageJ')
    class ImageJPlus(object):
        @property
        def py(self):
            return imagejPythonObj

        @property
        def legacy(self):
            if not hasattr(self, '_legacy'):
                try:
                    LegacyService = sj.jimport('net.imagej.legacy.LegacyService')
                    self._legacy = self.get('net.imagej.legacy.LegacyService')
                    if self.ui().isHeadless():
                        logging.warning("Operating in headless mode - the original ImageJ will have limited functionality.")
                except TypeError:
                    self._legacy = None

            if self._legacy is None:
                self._raise_legacy_missing_error()

            return self._legacy

        @property
        def IJ(self):
            return self._access_legacy_class('ij.IJ')

        @property
        def ResultsTable(self):
            return self._access_legacy_class('ij.measure.ResultsTable')

        @property
        def RoiManager(self):
            return self._access_legacy_class('ij.plugin.frame.RoiManager')

        @property
        def WindowManager(self):
            return self._access_legacy_class('ij.WindowManager')

        def _access_legacy_class(self, fqcn:str):
            self._check_legacy_active()
            class_name = fqcn[fqcn.rindex('.')+1:]
            property_name = f"_{class_name}"
            if not hasattr(self, property_name):
                if self.ui().isHeadless():
                    logging.warning(f"Operating in headless mode - the {class_name} class will not be fully functional.")
                setattr(self, property_name, sj.jimport(fqcn))
            
            return getattr(self, property_name)


        def _check_legacy_active(self):
            if not self.legacy or not self.legacy.isActive():
                self._raise_legacy_missing_error()

        def _raise_legacy_missing_error(self):
            raise ImportError("The original ImageJ is not available in this environment. Please include ImageJ Legacy in initialization. See: https://github.com/imagej/pyimagej/blob/master/doc/Initialization.md#how-to-initialize-imagej")

    # Overload operators for RandomAccessibleInterval so it's more Pythonic.
    @JImplementationFor('net.imglib2.RandomAccessibleInterval')
    class RAIOperators(object):
        def __add__(self, other):
            return ij.op().run('math.add', self, other)
        def __sub__(self, other):
            return ij.op().run('math.sub', self, other)
        def __mul__(self, other):
            return ij.op().run('math.mul', self, other)
        def __truediv__(self, other):
            return ij.op().run('math.div', self, other)
        def _slice(self, ranges):
            expected_dims = len(ranges)
            actual_dims = self.numDimensions()
            if expected_dims != actual_dims:
                raise ValueError(f'Dimension mismatch: {expected_dims} != {actual_dims}')

            imin = []
            imax = []
            for dslice in ranges:
                if dslice.step and dslice.step != 1:
                    raise ValueError(f'Unsupported step value: {dslice.step}')
                imin.append(dslice.start)
                imax.append(dslice.stop - 1)

            # BE WARNED! This does not yet preserve net.imagej-level axis metadata!
            # We need to finish RichImg to support that properly.
            return ij.op().transform().intervalView(self, imin, imax)

        def __getitem__(self, key):
            if type(key) == slice:
                # Wrap single slice into tuple of length 1.
                return self._slice((key,))
            elif type(key) == tuple:
                return self._slice(key)
            elif type(key) == int:
                return self.values[key]
            else:
                raise ValueError(f"Invalid key type: {type(key)}")

        def squeeze(self, axis=None):
            if axis is None:
                # Process all dimensions.
                axis = tuple(range(self.numDimensions()))
            if type(axis) == int:
                # Convert int to singleton tuple.
                axis = (axis,)
            if type(axis) != tuple:
                raise ValueError(f'Invalid type for axis parameter: {type(axis)}')

            Views = sj.jimport('net.imglib2.view.Views')
            res = self
            for d in range(self.numDimensions() - 1, -1, -1):
                if d in axis and self.dimension(d) == 1:
                    res = Views.hyperSlice(res, d, self.min(d));
            return res

    # Forward stdout and stderr from Java to Python.
    from jpype import JOverride, JImplements
    @JImplements('org.scijava.console.OutputListener')
    class JavaOutputListener():

        @JOverride
        def outputOccurred(self, e):
            source = e.getSource().toString
            output = e.getOutput()
            if source == 'STDOUT':
                sys.stdout.write(output)
            elif source == 'STDERR':
                sys.stderr.write(output)
            else:
                sys.stderr.write('[{}] {}'.format(source, output))

    ij.py._outputMapper = JavaOutputListener()
    ij.console().addOutputListener(ij.py._outputMapper)

    return ij


def imagej_main():
    args = []
    for i in range(1, len(sys.argv)):
        args.append(sys.argv[i])
    ij = init(headless='--headless' in args)
    # TODO: Investigate why ij.launch(args) doesn't work.
    ij.ui().showUI()
