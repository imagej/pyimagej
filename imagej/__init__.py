"""
PyImageJ provides a set of wrapper functions for integration between ImageJ
and Python. A major advantage of this approach is the ability to combine
ImageJ with other tools available from the Python software ecosystem,
including NumPy, SciPy, scikit-image, CellProfiler, OpenCV, ITK and more.

The first step when using pyimagej is to create an ImageJ2 gateway.
This gateway can point to any official release of ImageJ2 or to a local
installation. Using the gateway, you have full access to the ImageJ2 API,
plus utility functions for translating between Python (NumPy, xarray,
pandas, etc.) and Java (ImageJ, ImageJ2, ImgLib2, etc.) structures.

Here is an example of opening an image using ImageJ2 and displaying it:

    # Create an ImageJ2 gateway with the newest available version of ImageJ2.
    import imagej
    ij = imagej.init()

    # Load an image.
    image_url = 'https://imagej.net/images/clown.png'
    jimage = ij.io().open(image_url)

    # Convert the image from ImageJ to xarray, a package that adds
    # labeled datasets to numpy (http://xarray.pydata.org/en/stable/).
    image = ij.py.from_java(jimage)

    # Display the image (backed by matplotlib).
    ij.py.show(image, cmap='gray')
"""

import logging
from math import perm
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

from enum import Enum
from pathlib import Path
from typing import List, Tuple

from jpype import JArray, JException, JImplementationFor, JObject, setupGuiEnvironment

from .config import __author__, __version__

_logger = logging.getLogger(__name__)

# Enable debug logging if DEBUG environment variable is set.
try:
    debug = os.environ['DEBUG']
    if debug:
        _logger.setLevel(logging.DEBUG)
except KeyError as e:
    pass


class Mode(Enum):
    """
    An environment mode for the ImageJ2 gateway.
    See the imagej.init function for details.
    """
    GUI = "gui"
    HEADLESS = "headless"
    INTERACTIVE = "interactive"

    def __eq__(self, other):
        return super() == other or self.value == other


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


def init(ij_dir_or_version_or_endpoint=None, mode=Mode.HEADLESS, add_legacy=True, headless=None):
    """Initialize an ImageJ2 environment.

    The environment can wrap a local ImageJ2 installation, or consist of a
    specific version of ImageJ2 downloaded on demand, or even an explicit list
    of Maven artifacts. The environment can be initialized in headless mode or
    GUI mode, and with or without support for the original ImageJ.
    Note: some original ImageJ operations do not function in headless mode.

    :param ij_dir_or_version_or_endpoint:
        Path to a local ImageJ2 installation (e.g. /Applications/Fiji.app),
        OR version of net.imagej:imagej artifact to launch (e.g. 2.3.0),
        OR endpoint of another artifact built on ImageJ2 (e.g. sc.fiji:fiji),
        OR list of Maven artifacts to include (e.g.
           ['net.imagej:imagej:2.3.0', 'net.imagej:imagej-legacy', 'net.preibisch:BigStitcher']).
        The default is the latest version of net.imagej:imagej.
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
        ij = imagej.init('sc.fiji:fiji', mode=imagej.Mode.GUI)
    """
    if headless is not None:
        logging.warning("The headless flag of imagej.init is deprecated. Use the mode argument instead.")
        mode = Mode.HEADLESS if headless else Mode.INTERACTIVE

    macos = sys.platform == 'darwin';

    if macos and mode == Mode.INTERACTIVE:
        raise EnvironmentError("Sorry, the interactive mode is not available on macOS.")

    if not sj.jvm_started():
        success = _create_jvm(ij_dir_or_version_or_endpoint, mode, add_legacy)
        if not success:
            raise RuntimeError('Failed to create a JVM with the requested environment.')

    if mode == Mode.GUI:
        # Show the GUI and block.
        if macos:
            # NB: This will block the calling (main) thread forever!
            setupGuiEnvironment(lambda: _create_gateway().ui().showUI())
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


def _create_jvm(ij_dir_or_version_or_endpoint=None, mode=Mode.HEADLESS, add_legacy=True):
    """
    Ensures the JVM is properly initialized and ready to go,
    with requested settings.

    :return: True iff the JVM was successfully started.
             Note that this function returns False if a JVM is already running;
             to check for that situation, you can use scyjava.jvm_started().
    """

    # Check if JPype JVM is already running
    if sj.jvm_started():
        _logger.debug('The JVM is already running.')
        return False

    # Initialize configuration.
    if mode == Mode.HEADLESS:
        sj.config.add_option('-Djava.awt.headless=true')
    if hasattr(sj, 'jvm_version') and sj.jvm_version()[0] >= 9:
        # Disable illegal reflection access warnings.
        sj.config.add_option('--add-opens=java.base/java.lang=ALL-UNNAMED')
        sj.config.add_option('--add-opens=java.base/java.util=ALL-UNNAMED')

    # We want ImageJ's endpoints to come first, so these will be restored later
    original_endpoints = sj.config.endpoints.copy()
    sj.config.endpoints.clear()
    init_failed = False

    if ij_dir_or_version_or_endpoint is None:
        # Use latest release of ImageJ.
        _logger.debug('Using newest ImageJ release')
        sj.config.endpoints.append('net.imagej:imagej')

    elif isinstance(ij_dir_or_version_or_endpoint, list):
        # Assume that this is a list of Maven endpoints
        if any(item.startswith('net.imagej:imagej-legacy') for item in ij_dir_or_version_or_endpoint):
            add_legacy = False

        endpoint = '+'.join(ij_dir_or_version_or_endpoint)
        _logger.debug('List of Maven coordinates given: %s', ij_dir_or_version_or_endpoint)
        sj.config.endpoints.append(endpoint)

    elif os.path.isdir(ij_dir_or_version_or_endpoint):
        # Assume path to local ImageJ installation.
        add_legacy = False
        path = ij_dir_or_version_or_endpoint
        # Adjust the CWD to the ImageJ app directory
        os.chdir(path)
        _logger.debug('Local path to ImageJ installation given: %s', path)
        num_jars = _set_ij_env(path)
        if num_jars <= 0:
            _logger.error('Given directory does not appear to be a valid ImageJ installation: %s', path)
            init_failed = True
        else:
            _logger.info("Added " + str(num_jars + 1) + " JARs to the Java classpath.")
            plugins_dir = str(Path(path, 'plugins'))
            jvm_options = '-Dplugins.dir=' + plugins_dir

    elif re.match('^(/|[A-Za-z]:)', ij_dir_or_version_or_endpoint):
        # Looks like a file path was intended, but it's not a folder.
        path = ij_dir_or_version_or_endpoint
        _logger.error('Local path given is not a directory: %s', path)
        init_failed = True

    elif ':' in ij_dir_or_version_or_endpoint:
        # Assume endpoint of an artifact.
        # Strip out white spaces
        endpoint = ij_dir_or_version_or_endpoint.replace("    ", "")
        if any(item.startswith('net.imagej:imagej-legacy') for item in endpoint.split('+')):
            add_legacy = False
        _logger.debug('Maven coordinate given: %s', endpoint)
        sj.config.endpoints.append(endpoint)

    else:
        # Assume version of net.imagej:imagej.
        version = ij_dir_or_version_or_endpoint
        # Skip ignore
        if not re.match('\\d+\\.\\d+\\.\\d+', version):
            _logger.error('Invalid initialization string: %s', version)
            init_failed = True
            return False
        else:
            _logger.debug('ImageJ version given: %s', version)
            sj.config.endpoints.append('net.imagej:imagej:' + version)

    if init_failed:
        # Restore any pre-existing endpoints to allow for re-initialization
        sj.config.endpoints.clear()
        sj.config.endpoints.extend(original_endpoints)
        return False

    if add_legacy:
        sj.config.endpoints.append('net.imagej:imagej-legacy:MANAGED')

    # Restore any pre-existing endpoints, after ImageJ's
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
            if "'dependencies.dependency.version' for net.imagej:imagej-legacy:jar is missing." in l:
                unmanaged_legacy = True
        if unmanaged_legacy:
            _logger.error("""
***Invalid Initialization: you may be using primary endpoint that lacks pom-scijava as a parent***
   To keep all ImageJ components at compatible versions we recommend using a primary endpoint with a pom-scijava parent.
   For example, by putting 'net.imagej:imagej' first in your list of endpoints.
   If you are sure you DO NOT want a primary endpoint with a pom-scijava parent, please re-initialize with 'add_legacy=False'.
""")
        sj.config.endpoints.clear()
        sj.config.endpoints.extend(original_endpoints)
        return False

    return True


def _create_gateway():
    JObjectArray = JArray(JObject)

    # Initialize ImageJ
    try:
        ImageJ = sj.jimport('net.imagej.ImageJ')
    except TypeError:
        _logger.error("""
***Invalid initialization: ImageJ was not found***
   Please update your initialization call to include an ImageJ application or endpoint (e.g. net.imagej:imagej).
   NOTE: You MUST restart your python interpreter as Java can only be started once.
""")
        return False

    global ij
    ij = ImageJ()

    # Append some useful utility functions to the ImageJ gateway.
    Dataset                  = sj.jimport('net.imagej.Dataset')
    ImgPlus                  = sj.jimport('net.imagej.ImgPlus')
    Img                      = sj.jimport('net.imglib2.img.Img')
    ImgView                  = sj.jimport('net.imglib2.img.ImgView')
    RandomAccessibleInterval = sj.jimport('net.imglib2.RandomAccessibleInterval')

    try:
        ImagePlus = sj.jimport('ij.ImagePlus')
    except TypeError:
        # No original ImageJ on the classpath.
        ImagePlus = None

    class ImageJPython:
        def __init__(self, ij):
            self._ij = ij

        def dims(self, image):
            """
            ij.py.dims() is deprecated.
            Import the 'dims' module and use dims.get_dims().

            :example:
                >>> import imagej.dims as dims
                >>> dims.get_dims(image)
            """
            logging.warning("ij.py.dims() is deprecated. Import the 'dims' module and use dims.get_dims().")
            if self._is_arraylike(image):
                return image.shape
            if not sj.isjava(image):
                raise TypeError('Unsupported type: ' + str(type(image)))
            if sj.jclass('net.imglib2.Dimensions').isInstance(image):
                return list(image.dimensionsAsLongArray())
            if ImagePlus and isinstance(image, ImagePlus):
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
            if RandomAccessibleInterval.class_.isInstance(image_or_type):
                Util = sj.jimport('net.imglib2.util.Util')
                ij2_type = Util.getTypeFromInterval(image_or_type)
                return self.dtype(ij2_type)

            # -- Original ImageJ images --
            if ImagePlus and isinstance(image_or_type, ImagePlus):
                ij1_type = image_or_type.getType()
                ij1_types = {
                    ImagePlus.GRAY8:  'uint8',
                    ImagePlus.GRAY16: 'uint16',
                    ImagePlus.GRAY32: 'float32', # NB: ImageJ's 32-bit type is float32, not uint32.
                }
                for t in ij1_types:
                    if ij1_type == t:
                        return np.dtype(ij1_types[t])
                raise TypeError('Unsupported original ImageJ type: {}'.format(ij1_type))

            raise TypeError('Unsupported Java type: ' + str(sj.jclass(image_or_type).getName()))

        def new_numpy_image(self, image):
            """
            ij.py.new_numpy_image() is deprecated.
            Use ij.py.initialize_numpy_image() instead.
            """
            try:
                dtype_to_use = self.dtype(image)
            except TypeError:
                dtype_to_use = np.dtype('float64')
            logging.warning("ij.py.new_numpy_image() is deprecated. Use ij.py.initialize_numpy_image() instead.")
            return np.zeros(self.dims(image), dtype=dtype_to_use)

        def initialize_numpy_image(self, rai: RandomAccessibleInterval) -> np.ndarray:
            """Initialize a numpy array with zeros and shape of the input RandomAccessibleInterval.

            Initialize a new numpy array with the same dtype and shape as the input
            RandomAccessibleInterval with zeros.

            :param rai: A RandomAccessibleInterval
            :return:
                A numpy array with the same dtype and shape as the input 
                RandomAccessibleInterval, filled with zeros.
            """
            try:
                dtype_to_use = self.dtype(rai)
            except TypeError:
                dtype_to_use = np.dtype('float64')

            # get shape of rai and invert
            shape = dims.get_shape(rai)
            shape.reverse()
            return np.zeros(shape, dtype=dtype_to_use)

        def rai_to_numpy(self, rai: RandomAccessibleInterval, numpy_array: np.ndarray) -> np.ndarray:
            """Copy a RandomAccessibleInterval into a numpy array.

            The input RandomAccessibleInterval is copied into the pre-initialized numpy array
            with either (1) "fast copy" via 'net.imagej.util.Images.copy' if available or
            the slower "copy.rai" method. Note that the input RandomAccessibleInterval and
            numpy array must have reversed dimensions relative to each other (e.g. [t, z, y, x, c] and [c, x, y, z, t]).
            Use _permute_rai_to_python() on the RandomAccessibleInterval to reorganize the dimensions.

            :param rai: A RandomAccessibleInterval ('net.imglib2.RandomAccessibleInterval').
            :return: A numpy array of the input RandomAccessibleInterval.
            """
            # check imagej-common version for fast copy availability.
            ijc_slow_copy_version = '0.30.0'
            ijc_active_version = sj.get_version(Dataset)
            fast_copy_available = sj.compare_version(ijc_slow_copy_version, ijc_active_version)

            if fast_copy_available:
                Images = sj.jimport('net.imagej.util.Images')
                Images.copy(rai, self.to_java(numpy_array))
            else:
                self._ij.op().run("copy.rai", self.to_java(numpy_array), rai)

            return numpy_array

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
            ij._check_legacy_active('Use of original ImageJ macros is not possible.')

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
            :return: A Java map of output names and values, key: value pais.

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

            Converts Python objects (e.g. 'xarray.DataArray') into the Java
            equivalents. For numpy arrays, the Java image points to the Python array.

            :param data: Python object to be converted into its respective Java counterpart.
            :return: A Java object convrted from Python.
            """
            if self._is_memoryarraylike(data):
                return self.to_img(data)
            if self._is_xarraylike(data):
                return self.to_dataset(data)
            return sj.to_java(data)

        def to_dataset(self, data):
            """Converts the data into an ImageJ dataset

            Converts a Python image object (e.g 'xarray.DataArray') into a 'net.imagej.Dataset' Java
            object.

            :param data: Python image object to be converted to Dataset.
            :return: A 'net.imagej.Dataset'.
            """
            if self._is_xarraylike(data):
                return self._xarray_to_dataset(data)
            if self._is_arraylike(data):
                return self._numpy_to_dataset(data)
            if sj.isjava(data):
                return self._java_to_dataset(data)

            raise TypeError(f'Type not supported: {type(data)}')

        def to_img(self, data):
            """Converts the data into an ImageJ img

            Converts a Python image object (e.g 'xarray.DataArray') into a 'net.imglib2.Img' Java
            object.

            :param data: Python image object to be converted to Dataset.
            :return: A 'net.imglib2.Img'.
            """
            if self._is_xarraylike(data):
                return self._xarray_to_img(data)
            if self._is_arraylike(data):
                return self._numpy_to_img(data)
            if sj.isjava(data):
                return self._java_to_img(data)

            raise TypeError(f'Type not supported: {type(data)}')


        def jargs(self, *args):
            """Converts Python arguments into a Java Object[]

            Converts Python arguments into a Java Object[] (i.e.: array of Java
            objects). This is particularly useful in combination with ImageJ's
            various run functions, including ij.command().run(...),
            ij.module().run(...), ij.script().run(...), and ij.op().run(...).

            :param args: The Python arguments to wrap into an Object[].
            :return: A Java Object[]
            """
            return JObjectArray([self.to_java(arg) for arg in args])


        def _numpy_to_dataset(self, data):
            rai = imglyb.to_imglib(data)
            return self._java_to_dataset(rai)


        def _numpy_to_img(self, data):
            rai = imglyb.to_imglib(data)
            return self._java_to_img(rai)


        def _xarray_to_dataset(self, xarr):
            """
            Converts a xarray dataarray to a dataset, inverting C-style (slow axis first) to F-style (slow-axis last)
            :param xarr: Pass an xarray dataarray and turn into a dataset.
            :return: The dataset
            """
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
            if dims._ends_with_channel_axis(xarr):
                vals = np.moveaxis(xarr.values, -1, 0)
                return self._numpy_to_img(vals)
            else:
                return self._numpy_to_img(xarr.values)


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
                if self._ij.convert().supports(data, Img): # no dim info
                    img = self._ij.convert().convert(data, Img)
                    return self._ij.dataset().create(ImgPlus(img))
                if self._ij.convert().supports(data, RandomAccessibleInterval): # no dim info
                    rai = self._ij.convert().convert(data, RandomAccessibleInterval)
                    return self._ij.dataset().create(rai)
            except Exception as exc:
                _dump_exception(exc)
                raise exc
            raise TypeError('Cannot convert to dataset: ' + str(type(data)))

        def _java_to_img(self, data):
            """
            Converts the data into a ImageJ Img
            """
            # This try checking is necessary because the set of ImageJ converters is not complete.
            try:
                if self._ij.convert().supports(data, Img):
                    return self._ij.convert().convert(data, Img)
                if self._ij.convert().supports(data, RandomAccessibleInterval):
                    rai = self._ij.convert().convert(data, RandomAccessibleInterval)
                    # TODO: can we check for support on this convertion before the conversion on 839?
                    return ImgView.wrap(rai)
            except Exception as exc:
                _dump_exception(exc)
                raise exc
            raise TypeError('Cannot convert to img: ' + str(type(data)))

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
                if ImagePlus and isinstance(data, ImagePlus):
                    data = self._imageplus_to_imgplus(data)
                if self._ij.convert().supports(data, ImgPlus):
                    if dims._has_axis(data):
                        # HACK: Converter exists for ImagePlus -> Dataset, but not ImagePlus -> RAI.
                        data = self._ij.convert().convert(data, ImgPlus)
                        permuted_rai = self._permute_rai_to_python(data)
                        numpy_result = self.initialize_numpy_image(permuted_rai)
                        numpy_result = self.rai_to_numpy(permuted_rai, numpy_result)
                        return self._dataset_to_xarray(permuted_rai,numpy_result)
                    if self._ij.convert().supports(data, RandomAccessibleInterval):
                        rai = self._ij.convert().convert(data, RandomAccessibleInterval)
                        numpy_result = self.initialize_numpy_image(rai)
                        return self.rai_to_numpy(rai, numpy_result)
            except Exception as exc:
                _dump_exception(exc)
                raise exc
            return sj.to_python(data)


        def _dataset_to_xarray(self, permuted_rai: RandomAccessibleInterval, numpy_array: np.ndarray) -> xr.DataArray:
            """Wrap a numpy array with xarray and axes metadta from a RandomAccessibleInterval.

            Wraps a numpy array with the metadata from the source RandomAccessibleInterval 
            metadata (i.e. axes).

            :param permuted_rai: A RandomAccessibleInterval with axes (e.g. Dataset or ImgPlus).
            :param numpy_array: A np.ndarray to wrap with xarray.
            :return: xarray.DataArray with metadata/axes.
            """
            # get metadata
            xr_axes = dims.get_axes(permuted_rai)
            xr_dims = dims.get_dims(permuted_rai)
            xr_attrs = sj.to_python(permuted_rai.getProperties())
            # reverse axes and dims to match numpy_array
            xr_axes.reverse()
            xr_dims.reverse()
            xr_dims = dims._convert_dims(xr_dims, direction='python')
            xr_coords = dims._get_axes_coords(xr_axes, xr_dims, numpy_array.shape)
            return xr.DataArray(numpy_array, dims=xr_dims, coords=xr_coords, attrs=xr_attrs)

        
        def _permute_rai_to_python(self, rai: RandomAccessibleInterval) -> RandomAccessibleInterval:
            """Permute a RandomAccessibleInterval to the python reference order.

            Permute a RandomAccessibleInterval to the Python reference order of
            CXYZT (where dimensions exist). Note that this is reverse from the final array order of 
            TZYXC.

            :param rai: A RandomAccessibleInterval with axes.
            :return: A permuted RandomAccessibleInterval.
            """
            # get input rai metadata if it exists
            try:
                rai_metadata = rai.getProperties()
            except AttributeError:
                rai_metadata = None

            rai_axis_types = dims.get_axis_types(rai)

            # permute rai to specified order and transfer metadata
            permute_order = dims.prioritize_rai_axes_order(rai_axis_types, dims._python_rai_ref_order())
            permuted_rai = dims.reorganize(rai, permute_order)

            # add metadata to image if it exisits
            if rai_metadata != None:
                permuted_rai.getProperties().putAll(rai_metadata)

            return permuted_rai


        def _imageplus_to_imgplus(self, imp: ImagePlus) -> ImgPlus:
            if ImagePlus and isinstance(imp, ImagePlus):
                ds = self._ij.convert().convert(imp, Dataset)
                return ds.getImgPlus()
            raise ValueError('Input is not an ImagePlus')


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

            if ij.legacy and ij.legacy.isActive():
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
            """ Synchronize data between ImageJ and ImageJ2.

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
            # Don't sync if the ImagePlus is not linked back to a corresponding dataset
            if str(type(pixels)) == '<class \'jnius.ByteArray\'>':
                return

            stack.setPixels(pixels, imp.getCurrentSlice())

    # attach ImageJPython to imagej
    imagejPythonObj = ImageJPython(ij)

    @JImplementationFor('net.imagej.ImageJ')
    class ImageJPlus(object):
        @property
        def py(self):
            return imagejPythonObj

        @property
        def legacy(self):
            """
            Gets the ImageJ2 gateway's LegacyService, or None if original
            ImageJ support is not available in the current environment.
            """
            if not hasattr(self, '_legacy'):
                try:
                    LegacyService = sj.jimport('net.imagej.legacy.LegacyService')
                    self._legacy = self.get('net.imagej.legacy.LegacyService')
                    if self.ui().isHeadless():
                        logging.warning("Operating in headless mode - the original ImageJ will have limited functionality.")
                except TypeError:
                    self._legacy = None

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
            self._check_legacy_active(f'The {fqcn} class is not available.')
            class_name = fqcn[fqcn.rindex('.')+1:]
            property_name = f"_{class_name}"
            if not hasattr(self, property_name):
                if self.ui().isHeadless():
                    logging.warning(f"Operating in headless mode - the {class_name} class will not be fully functional.")
                setattr(self, property_name, sj.jimport(fqcn))

            return getattr(self, property_name)

        def _check_legacy_active(self, usage_context=''):
            if not self.legacy or not self.legacy.isActive():
                raise ImportError(f"The original ImageJ is not available in this environment. {usage_context} See: https://github.com/imagej/pyimagej/blob/master/doc/Initialization.md")

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
        def _index(self, position):
            ra = self.randomAccess()
            # Can we store this as a shape property?
            dims = ij.py.dims(self)
            for i in range(len(position)):
                pos = position[i] % dims[i]
                ra.setPosition(pos, i)
            # TODO: Are we assuming too much here with the RealType.get()
            return ra.get().get()
        def _is_index(self, a):
            # Check dimensionality - if we don't have enough dims, it's a slice
            num_dims = 1 if type(a) == int else len(a)
            if num_dims < self.numDimensions(): return False
            # if an int, it is an index
            if type(a) == int: return True
            # if we have a tuple, it's an index if there are any slices
            hasSlice = True in [type(item) == slice for item in a]
            return not hasSlice
        def _slice(self, ranges):
            expected_dims = len(ranges)
            actual_dims = self.numDimensions()
            if expected_dims > actual_dims:
                raise ValueError(f'Dimension mismatch: {expected_dims} > {actual_dims}')
            elif expected_dims < actual_dims:
                ranges = (list(ranges) + actual_dims * [slice(None)])[:actual_dims]

            imin = []
            imax = []
            dslices = [r if type(r) == slice else slice(r, r+1) for r in ranges]
            for dslice in dslices:
                if dslice.step and dslice.step != 1:
                    raise ValueError(f'Unsupported step value: {dslice.step}')
                imax.append(None if dslice.stop == None else dslice.stop - 1)
                imin.append(None if dslice.start == None else dslice.start)

            # BE WARNED! This does not yet preserve net.imagej-level axis metadata!
            # We need to finish RichImg to support that properly.

            return stack.rai_slice(self, tuple(imin), tuple(imax))


        def __getitem__(self, key):
            if type(key) == slice:
                # Wrap single slice into tuple of length 1.
                return self._slice((key,))
            elif type(key) == tuple:
                return self._index(key) if self._is_index(key) else self._slice(key)
            elif type(key) == int:
                # Wrap single int into tuple of length 1.
                return self.__getitem__((key, ))
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
                    res = Views.hyperSlice(res, d, self.min(d))
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

    sj.when_jvm_stops(lambda: ij.dispose())

    return ij


def imagej_main():
    args = []
    for i in range(1, len(sys.argv)):
        args.append(sys.argv[i])
    mode = 'headless' if '--headless' in args else 'gui'
    ij = init(mode=mode)
