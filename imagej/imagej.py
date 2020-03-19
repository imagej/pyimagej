"""
wrapper for imagej and python integration using ImgLyb

"""

# TODO: Unify version declaration to one place.
# https://www.python.org/dev/peps/pep-0396/#deriving
__version__ = '0.6.0.dev0'
__author__ = 'Curtis Rueden, Yang Liu, Michael Pinkert'

import logging, os, re, sys
import scyjava_config
import jnius_config
from pathlib import Path
import numpy
import xarray as xr
import warnings

_logger = logging.getLogger(__name__)

# Enable debug logging if DEBUG environment variable is set.
try:
    debug = os.environ['DEBUG']
    if debug:
        _logger.setLevel(logging.DEBUG)
except KeyError as e:
    pass


def _dump_exception(exc):
    if _logger.isEnabledFor(logging.DEBUG) and hasattr(exc, 'stacktrace'):
        _logger.debug("\n\tat ".join([str(e) for e in exc.stacktrace]))


def search_for_jars(ij_dir, subfolder):
    """
    Search and add .jars ile to a list
    :param ij_dir: System path for Fiji.app
    :param subfolder: the folder needs to be searched
    :return: a list of jar files
    """
    jars = []
    for root, dirs, files in os.walk(ij_dir + subfolder):
        for f in files:
            if f.endswith('.jar'):
                path = root + '/' + f
                jars.append(path)
                _logger.debug('Added %s', path)
    return jars


def set_ij_env(ij_dir):
    """
    Create a list of required jars and add to the java classpath

    :param ij_dir: System path for Fiji.app
    :return: num_jar(int): number of jars added
    """
    jars = []
    # search jars directory
    jars.extend(search_for_jars(ij_dir, '/jars'))
    # search plugins directory
    jars.extend(search_for_jars(ij_dir, '/plugins'))
    # add to classpath
    scyjava_config.add_classpath(os.pathsep.join(jars))
    return len(jars)


def init(ij_dir_or_version_or_endpoint=None, headless=True, new_instance=False):
    """
    Initialize the ImageJ environment.

    :param ij_dir_or_version_or_endpoint:
        Path to a local ImageJ installation (e.g. /Applications/Fiji.app),
        OR version of net.imagej:imagej artifact to launch (e.g. 2.0.0-rc-67),
        OR endpoint of another artifact (e.g. sc.fiji:fiji) that uses imagej.
    :param headless: Whether to start the JVM in headless or gui mode.
    :param new_instance: If JVM is already running, setting this parameter to
        True will create a new ImageJ instance.
    :return: an instance of the net.imagej.ImageJ gateway
    """

    global ij

    if jnius_config.vm_running and not new_instance:
        _logger.warning('The JVM is already running.')
        return ij

    if not jnius_config.vm_running:

        if headless:
            scyjava_config.add_options('-Djava.awt.headless=true')

        if ij_dir_or_version_or_endpoint is None:
            # Use latest release of ImageJ.
            _logger.debug('Using newest ImageJ release')
            scyjava_config.add_endpoints('net.imagej:imagej')

        elif os.path.isdir(ij_dir_or_version_or_endpoint):
            # Assume path to local ImageJ installation.
            path = ij_dir_or_version_or_endpoint
            _logger.debug('Local path to ImageJ installation given: %s', path)
            num_jars = set_ij_env(path)
            _logger.info("Added " + str(num_jars + 1) + " JARs to the Java classpath.")
            plugins_dir = str(Path(path, 'plugins'))
            scyjava_config.add_options('-Dplugins.dir=' + plugins_dir)

        elif re.match('^(/|[A-Za-z]:)', ij_dir_or_version_or_endpoint):
            # Looks like a file path was intended, but it's not a folder.
            path = ij_dir_or_version_or_endpoint
            _logger.error('Local path given is not a directory: %s', path)
            return False

        elif ':' in ij_dir_or_version_or_endpoint:
            # Assume endpoint of an artifact.
            endpoint = ij_dir_or_version_or_endpoint
            _logger.debug('Maven coordinate given: %s', endpoint)
            scyjava_config.add_endpoints(endpoint)

        else:
            # Assume version of net.imagej:imagej.
            version = ij_dir_or_version_or_endpoint
            _logger.debug('ImageJ version given: %s', version)
            scyjava_config.add_endpoints('net.imagej:imagej:' + version)

    # Must import imglyb (not scyjava) to spin up the JVM now.
    import imglyb
    from jnius import autoclass
    from jnius import cast
    import scyjava

    # Initialize ImageJ.
    ImageJ = autoclass('net.imagej.ImageJ')
    ij = ImageJ()

    # Append some useful utility functions to the ImageJ gateway.

    from scyjava import jclass, isjava, to_java, to_python

    Dataset                  = autoclass('net.imagej.Dataset')
    ImgPlus                  = autoclass('net.imagej.ImgPlus')
    Img                      = autoclass('net.imglib2.img.Img')
    RandomAccessibleInterval = autoclass('net.imglib2.RandomAccessibleInterval')
    Axes                     = autoclass('net.imagej.axis.Axes')
    DefaultLinearAxis        = autoclass('net.imagej.axis.DefaultLinearAxis')

    class ImageJPython:
        def __init__(self, ij):
            self._ij = ij

        def dims(self, image):
            """
            Return the dimensions of the equivalent numpy array for the image.  Reverse dimension order from Java.
            """
            if self._is_arraylike(image):
                return image.shape
            if not isjava(image):
                raise TypeError('Unsupported type: ' + str(type(image)))
            if jclass('net.imglib2.Dimensions').isInstance(image):
                return [image.dimension(d) for d in range(image.numDimensions() -1, -1, -1)]
            if jclass('ij.ImagePlus').isInstance(image):
                dims = image.getDimensions()
                dims.reverse()
                dims = [dim for dim in dims if dim > 1]
                return dims
            raise TypeError('Unsupported Java type: ' + str(jclass(image).getName()))

        def dtype(self, image_or_type):
            """
            Return the dtype of the equivalent numpy array for the given image or type.
            """
            if type(image_or_type) == numpy.dtype:
                return image_or_type
            if self._is_arraylike(image_or_type):
                return image_or_type.dtype
            if not isjava(image_or_type):
                raise TypeError('Unsupported type: ' + str(type(image_or_type)))

            # -- ImgLib2 types --
            if jclass('net.imglib2.type.Type').isInstance(image_or_type):
                ij2_types = {
                    'net.imglib2.type.logic.BitType':                     'bool',
                    'net.imglib2.type.numeric.integer.ByteType':          'int8',
                    'net.imglib2.type.numeric.integer.ShortType':         'int16',
                    'net.imglib2.type.numeric.integer.IntType':           'int32',
                    'net.imglib2.type.numeric.integer.LongType':          'int64',
                    'net.imglib2.type.numeric.integer.UnsignedByteType':  'uint8',
                    'net.imglib2.type.numeric.integer.UnsignedShortType': 'uint16',
                    'net.imglib2.type.numeric.integer.UnsignedIntType':   'uint32',
                    'net.imglib2.type.numeric.integer.UnsignedLongType':  'uint64',
                    'net.imglib2.type.numeric.real.FloatType':            'float32',
                    'net.imglib2.type.numeric.real.DoubleType':           'float64',
                }
                for c in ij2_types:
                    if jclass(c).isInstance(image_or_type):
                        return numpy.dtype(ij2_types[c])
                raise TypeError('Unsupported ImgLib2 type: {}'.format(image_or_type))

            # -- ImgLib2 images --
            if jclass('net.imglib2.IterableInterval').isInstance(image_or_type):
                ij2_type = image_or_type.firstElement()
                return self.dtype(ij2_type)
            if jclass('net.imglib2.RandomAccessibleInterval').isInstance(image_or_type):
                Util = autoclass('net.imglib2.util.Util')
                ij2_type = Util.getTypeFromInterval(image_or_type)
                return self.dtype(ij2_type)

            # -- ImageJ1 images --
            if jclass('ij.ImagePlus').isInstance(image_or_type):
                ij1_type = image_or_type.getType()
                ImagePlus = autoclass('ij.ImagePlus')
                ij1_types = {
                    ImagePlus.GRAY8:  'uint8',
                    ImagePlus.GRAY16: 'uint16',
                    ImagePlus.GRAY32: 'float32', # NB: ImageJ1's 32-bit type is float32, not uint32.
                }
                for t in ij1_types:
                    if ij1_type == t:
                        return numpy.dtype(ij1_types[c])
                raise TypeError('Unsupported ImageJ1 type: {}'.format(ij1_type))

            raise TypeError('Unsupported Java type: ' + str(jclass(image_or_type).getName()))

        def new_numpy_image(self, image):
            """
            Creates a numpy image (NOT a Java image) dimensioned the same as
            the given image, and with the same pixel type as the given image.
            """
            try:
                dtype_to_use = self.dtype(image)
            except TypeError:
                dtype_to_use = numpy.dtype('float64')
            return numpy.zeros(self.dims(image), dtype=dtype_to_use)

        def rai_to_numpy(self, rai):
            """
            Convert a RandomAccessibleInterval into a numpy array
            """
            result = self.new_numpy_image(rai)
            self._ij.op().run("copy.rai", self.to_java(result), rai)
            return result

        def run_plugin(self, plugin, args=None, ij1_style=True):
            """
            Run an ImageJ plugin
            :param plugin: The string name for the plugin command
            :param args: A dict of macro arguments in key/value pairs
            :param ij1_style: Whether to use implicit booleans in IJ1 style or explicit booleans in IJ2 style
            :return: The plugin output
            """
            macro = self._assemble_plugin_macro(plugin, args=args, ij1_style=ij1_style)
            return self.run_macro(macro)

        def run_macro(self, macro, args=None):
            """
            Run an ImageJ1 style macro script
            :param macro: The macro code
            :param args: Arguments for the script as a dictionary of key/value pairs
            :return:
            """
            try:
                if args is None:
                    return self._ij.script().run("macro.ijm", macro, True).get()
                else:
                    return self._ij.script().run("macro.ijm", macro, True, to_java(args)).get()
            except Exception as exc:
                _dump_exception(exc)
                raise exc

        def run_script(self, language, script, args=None):
            """
            Run a script in an IJ scripting language
            :param language: The file extension for the scripting language
            :param script: A string containing the script code
            :param args: Arguments for the script as a dictionary of key/value pairs
            :return:
            """
            script_lang = self._ij.script().getLanguageByName(language)
            if script_lang is None:
                script_lang = self._ij.script().getLanguageByExtension(language)
            if script_lang is None:
                raise ValueError("Unknown script language: " + language)
            exts = script_lang.getExtensions()
            if exts.isEmpty():
                raise ValueError("Script language '" + script_lang.getLanguageName() + "' has no extensions")
            ext = exts.get(0)
            try:
                if args is None:
                    return self._ij.script().run("script." + ext, script, True).get()
                return self._ij.script().run("script." + ext, script, True, to_java(args)).get()
            except Exception as exc:
                _dump_exception(exc)
                raise exc

        def to_java(self, data):
            """
            Converts the data into a java equivalent.  For numpy arrays, the java image points to the python array.

            In addition to the scyjava types, we allow ndarray-like and xarray-like variables
            """
            if self._is_memoryarraylike(data):
                return imglyb.to_imglib(data)
            if self._is_xarraylike(data):
                return self.to_dataset(data)
            return to_java(data)

        def to_dataset(self, data):
            """Converts the data into an ImageJ dataset"""
            if self._is_xarraylike(data):
                return self._xarray_to_dataset(data)
            if self._is_arraylike(data):
                return self._numpy_to_dataset(data)
            if scyjava.isjava(data):
                return self._java_to_dataset(data)

            raise TypeError(f'Type not supported: {type(data)}')

        def _numpy_to_dataset(self, data):
            rai = imglyb.to_imglib(data)
            return self._java_to_dataset(rai)

        def _xarray_to_dataset(self, xarr):
            """
            Converts a xarray dataarray with specified dim order to an image
            :param xarr: Pass an xarray dataarray and turn into a dataset.
            :return: The dataset
            """
            dataset = self._numpy_to_dataset(xarr.values)
            axes = self._assign_axes(xarr)
            dataset.setAxes(axes)

            self._assign_dataset_metadata(dataset, xarr.attrs)

            return dataset
        
        def _assign_axes(self, xarr):
            """
            Obtain xarray axes names, origin, and scale and convert into ImageJ Axis; currently supports DefaultLinearAxis.
            :param xarr: xarray that holds the units
            :return: A list of ImageJ Axis with the specified origin and scale
            """
            axes = ['']*len(xarr.dims)

            for axis in xarr.dims:
                origin = self._get_origin(xarr.coords[axis])
                scale = self._get_scale(xarr.coords[axis])

                axisStr = self._pydim_to_ijdim(axis)

                ax_type = Axes.get(axisStr)
                ax_num = self._get_axis_num(xarr, axis)
                if scale is None:
                    java_axis = DefaultLinearAxis(ax_type)
                else:
                    java_axis = DefaultLinearAxis(ax_type, numpy.double(scale), numpy.double(origin))

                axes[ax_num] = java_axis

            return axes

        def _pydim_to_ijdim(self, axis):
            """Convert between the lowercase Python convention (x, y, z, c, t) to IJ (X, Y, Z, C, T)"""
            if str(axis) in ['x', 'y', 'z', 'c', 't']:
                return str(axis).upper()
            return str(axis)

        def _ijdim_to_pydim(self, axis):
            """Convert the IJ uppercase dimension convention (X, Y, Z< C, T) to lowercase python (x, y, z, c, t) """
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
            if numpy.isfortran(xarr.values):
                return py_axnum

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
                    return self._ij.dataset().create(rai)
            except Exception as exc:
                _dump_exception(exc)
                raise exc
            raise TypeError('Cannot convert to dataset: ' + str(type(data)))

        def from_java(self, data):
            """
            Converts the data into a python equivalent
            """
            # todo: convert a datset to xarray

            if not isjava(data): return data
            try:
                if self._ij.convert().supports(data, Dataset):
                    # HACK: Converter exists for ImagePlus -> Dataset, but not ImagePlus -> RAI.
                    data = self._ij.convert().convert(data, Dataset)
                    return self._dataset_to_xarray(data)
                if (self._ij.convert().supports(data, RandomAccessibleInterval)):
                    rai = self._ij.convert().convert(data, RandomAccessibleInterval)
                    return self.rai_to_numpy(rai)
            except Exception as exc:
                _dump_exception(exc)
                raise exc
            return to_python(data)

        def _dataset_to_xarray(self, dataset):
            """
            Converts an ImageJ dataset into an xarray
            :param dataset: ImageJ dataset
            :return: xarray with reversed (C-style) dims and coords as labeled by the dataset
            """
            attrs = self._ij.py.from_java(dataset.getProperties())
            axes = [(cast('net.imagej.axis.DefaultLinearAxis', dataset.axis(idx)))
                    for idx in range(dataset.numDimensions())]

            dims = [self._ijdim_to_pydim(axes[idx].type().getLabel()) for idx in range(len(axes))]
            values = self.rai_to_numpy(dataset)
            coords = self._get_axes_coords(axes, dims, numpy.shape(numpy.transpose(values)))

            xarr = xr.DataArray(values, dims=list(reversed(dims)), coords=coords, attrs=attrs)
            return xarr

        def _get_axes_coords(self, axes, dims, shape):
            """
            Get xarray style coordinate list dictionary from a dataset
            :param axes: List of ImageJ axes
            :param dims: List of axes labels for each dataset axis
            :param shape: F-style, or reversed C-style, shape of axes numpy array.
            :return: Dictionary of coordinates for each axis.
            """
            coords = {dims[idx]: numpy.arange(axes[idx].origin(), shape[idx]*axes[idx].scale() + axes[idx].origin(),
                                              axes[idx].scale())
                      for idx in range(len(dims))}
            return coords


        def show(self, image, cmap=None):
            """
            Display a java or python 2D image.
            :param image: A java or python image that can be converted to a numpy array
            :param cmap: The colormap of the image, if it is not RGB
            :return:
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
            temp_value = str(value).replace('\\', '/')
            if temp_value.startswith('[') and temp_value.endswith(']'):
                    return temp_value
            final_value = '[' + temp_value + ']'
            return final_value

    ij.py = ImageJPython(ij)

    # Forward stdout and stderr from Java to Python.

    from jnius import PythonJavaClass, java_method

    class JavaOutputListener(PythonJavaClass):
        __javainterfaces__ = ['org/scijava/console/OutputListener']

        @java_method('(Lorg/scijava/console/OutputEvent;)V')
        def outputOccurred(self, e):
            source = e.getSource().toString()
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


def help():
    """
    print the instruction for using imagej module

    :return:
    """

    print(("Please set the environment variables first:\n"
           "Fiji.app:   ij_dir = 'your local fiji.app path'\n"
           "Then call init(ij_dir)"))
