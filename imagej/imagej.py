"""
wrapper for imagej and python integration using ImgLyb

"""

# TODO: Unify version declaration to one place.
# https://www.python.org/dev/peps/pep-0396/#deriving
__version__ = '0.6.0.dev0'
__author__ = 'Curtis Rueden, Yang Liu, Michael Pinkert'

import logging, os, re, sys
import scyjava_config
import jpype
import jpype.imports
import imglyb
import scyjava
from pathlib import Path
import numpy
import xarray as xr

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
    Search and add .jar files to a list
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
    jpype.addClassPath(os.pathsep.join(jars))
    print('jpype classpath: {0}'.format(jpype.getClassPath()))
    scyjava_config.add_classpath(os.pathsep.join(jars))
    return len(jars)


def init(ij_dir_or_version_or_endpoint=None, headless=True, new_instance=False):
    """
    Initialize the ImageJ environment.

    :param ij_dir_or_version_or_endpoint:
        Path to a local ImageJ installation (e.g. /Applications/Fiji.app),
        OR version of net.imagej:imagej artifact to launch (e.g. 2.0.0-rc-67),
        OR endpoint of another artifact (e.g. sc.fiji:fiji) that uses imagej.
        OR list of Maven artifacts to include (e.g. ['net.imagej:imagej-legacy', 'net.preibisch:BigStitcher'])
    :param headless: Whether to start the JVM in headless or gui mode.
    :param new_instance: If JVM is already running, setting this parameter to
        True will create a new ImageJ instance.
    :return: an instance of the net.imagej.ImageJ gateway
    """

    global ij    

    # EE: Check if JPype JVM is already running
    jvm_status = jpype.isJVMStarted()
    if jvm_status == True:
        print('The JPype JVM is already running.')

    #if jnius_config.vm_running and not new_instance:
    #    _logger.warning('The JVM is already running.')
    #    return ij

    ## EE: Configure the JPype JVM
    
    if not jvm_status:

        if headless:
            jvm_options = '-Djava.awt.headless=true'

        if ij_dir_or_version_or_endpoint is None:
            # Use latest release of ImageJ.
            _logger.debug('Using newest ImageJ release')
            scyjava_config.add_endpoints('net.imagej:imagej')

        elif isinstance(ij_dir_or_version_or_endpoint, list):
            # Assume that this is a list of Maven endpoints
            endpoint = '+'.join(ij_dir_or_version_or_endpoint)
            _logger.debug('List of Maven coordinates given: %s', ij_dir_or_version_or_endpoint)
            scyjava_config.add_endpoints(endpoint)

        elif os.path.isdir(ij_dir_or_version_or_endpoint):
            # Assume path to local ImageJ installation.
            path = ij_dir_or_version_or_endpoint
            _logger.debug('Local path to ImageJ installation given: %s', path)
            num_jars = set_ij_env(path)
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
            endpoint = ij_dir_or_version_or_endpoint.replace(" ", "")
            _logger.debug('Maven coordinate given: %s', endpoint)
            scyjava_config.add_endpoints(endpoint)

        else:
            # Assume version of net.imagej:imagej.
            version = ij_dir_or_version_or_endpoint
            _logger.debug('ImageJ version given: %s', version)
            scyjava_config.add_endpoints('net.imagej:imagej:' + version)

    # Start jpype jvm here?
    # Must import imglyb (not scyjava) to spin up the JVM now.
    #from jnius import autoclass, JavaException, cast

    # Initialize ImageJ.
    ImageJ = jpype.JClass('net.imagej.ImageJ')
    ij = ImageJ()

    # Append some useful utility functions to the ImageJ gateway.

    from scyjava import jclass, isjava, to_java, to_python

    Dataset                  = jpype.JClass('net.imagej.Dataset')
    ImgPlus                  = jpype.JClass('net.imagej.ImgPlus')
    Img                      = jpype.JClass('net.imglib2.img.Img')
    RandomAccessibleInterval = jpype.JClass('net.imglib2.RandomAccessibleInterval')
    Axes                     = jpype.JClass('net.imagej.axis.Axes')
    Double                   = jpype.JClass('java.lang.Double')

    # EnumeratedAxis is a new axis made for xarray, so is only present in ImageJ versions that are released
    # later than March 2020.  This check defaults to LinearAxis instead if Enumerated does not work.
    try:
        EnumeratedAxis           = jpype.JClass('net.imagej.axis.EnumeratedAxis')
    except JavaException:
        DefaultLinearAxis = jpype.JClass('net.imagej.axis.DefaultLinearAxis')
        def EnumeratedAxis(axis_type, values):
            origin = values[0]
            scale = values[1] - values[0]
            axis = DefaultLinearAxis(axis_type, scale, origin)
            return axis

    # Try to define the legacy service, and create a dummy method if it doesn't exist.
    try:
        LegacyService = jpype.JClass('net.imagej.legacy.LegacyService')
        legacyService = cast(LegacyService, ij.get("net.imagej.legacy.LegacyService"))
    except JavaException:
        class LegacyService:
            def isActive(self):
                return False
        legacyService = LegacyService()

    # Create a method to get the legacy service that is similar to other ImageJ services
    def legacy():
        try:
            legacyService = cast(LegacyService, ij.get('net.imagej.legacy.LegacyService'))
        except JavaException:
            legacyService = LegacyService()
        return legacyService
    setattr(ij, 'legacy', legacy)

    if legacyService.isActive():
            WindowManager = jpype.JClass('ij.WindowManager')
    else:
        class WindowManager:
            def getCurrentImage(self):
                """
                Throw an error saying IJ1 is not available
                :return:
                """
                raise ImportError("Your ImageJ installation does not support IJ1.  This function does not work.")
        WindowManager = WindowManager()

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
                        return numpy.dtype(ij1_types[t])
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
            if not ij.legacy().isActive():
                raise ImportError("Your IJ endpoint does not support IJ1, and thus cannot use IJ1 macros.")

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
                vals = numpy.moveaxis(xarr.values, -1, 0)
                dataset = self._numpy_to_dataset(vals)
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
                    doub_coords = [Double(numpy.double(x)) for x in numpy.arange(len(xarr.coords[axis]))]
                else:
                    doub_coords = [Double(numpy.double(x)) for x in xarr.coords[axis]]

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
            if numpy.isfortran(xarr.values):
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
                if self._ij.convert().supports(data, RandomAccessibleInterval):
                    rai = self._ij.convert().convert(data, RandomAccessibleInterval)
                    return self.rai_to_numpy(rai)
            except Exception as exc:
                _dump_exception(exc)
                raise exc
            return to_python(data)

        def _dataset_to_xarray(self, dataset):
            """
            Converts an ImageJ dataset into an xarray, inverting F-style (slow idx last) to C-style (slow idx first)
            :param dataset: ImageJ dataset
            :return: xarray with reversed (C-style) dims and coords as labeled by the dataset
            """
            attrs = self._ij.py.from_java(dataset.getProperties())
            axes = [(cast('net.imagej.axis.CalibratedAxis', dataset.axis(idx)))
                    for idx in range(dataset.numDimensions())]

            dims = [self._ijdim_to_pydim(axes[idx].type().getLabel()) for idx in range(len(axes))]
            values = self.rai_to_numpy(dataset)
            coords = self._get_axes_coords(axes, dims, numpy.shape(numpy.transpose(values)))

            if dims[len(dims)-1].lower() in ['c', 'channel']:
                xarr_dims = self._invert_except_last_element(dims)
                values = numpy.moveaxis(values, 0, -1)
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

        def window_manager(self):
            """
            Get the ImageJ1 window manager if legacy mode is enabled.  It may not work properly if in headless mode.
            :return: WindowManager
            """
            if not ij.legacy_enabled:
                raise ImportError("Your ImageJ installation does not support IJ1.  This function does not work.")
            elif ij.ui().isHeadless():
                logging.warning("Operating in headless mode - The WindowManager will not be fully funtional.")
            else:
                return WindowManager

        def active_xarray(self, sync=True):
            """
            Convert the active image to a xarray.DataArray, synchronizing from IJ1 -> IJ2
            :param sync: Manually synchronize the current IJ1 slice if True
            :return: numpy array containing the image data
            """
            # todo: make the behavior use pure IJ2 if legacy is not active

            if ij.legacy().isActive():
                imp = self.active_image_plus(sync=sync)
                return self._ij.py.from_java(imp)
            else:
                dataset = self.active_dataset()
                return self._ij.py.from_java(dataset)

        def active_dataset(self):
            """Get the currently active Dataset from the Dataset service"""
            return self._ij.imageDisplay().getActiveDataset()

        def active_image_plus(self, sync=True):
            """
            Get the currently active IJ1 image, optionally synchronizing from IJ1 -> IJ2
            :param sync: Manually synchronize the current IJ1 slice if True
            :return: The ImagePlus corresponding to the active image
            """
            imp = WindowManager.getCurrentImage()
            if sync:
                self.synchronize_ij1_to_ij2(imp)
            return imp

        def synchronize_ij1_to_ij2(self, imp):
            """
            Synchronize between a Dataset or ImageDisplay linked to an ImagePlus by accepting the ImagePlus data as true
            :param imp: The IJ1 ImagePlus that needs to be synchronized
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
