"""
wrapper for imagej and python integration using ImgLyb

"""

# TODO: Unify version declaration to one place.
# https://www.python.org/dev/peps/pep-0396/#deriving
__version__ = '0.4.0.dev0'
__author__ = 'Curtis Rueden & Yang Liu'

import os
import scyjava_config
from pathlib import Path
import numpy


def _debug(message):
    """
    print debug message

    :param message: Debug message to be printed
    :return: None
    """
    if not __debug__:
        print(message)


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
                _debug('Added ' + path)
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


def init(ij_dir_or_version_or_endpoint=None, headless=True):
    """
    Initialize the ImageJ environment.

    :param ij_dir_or_version_or_endpoint:
        Path to a local ImageJ installation (e.g. /Applications/Fiji.app),
        OR version of net.imagej:imagej artifact to launch (e.g. 2.0.0-rc-67),
        OR endpoint of another artifact (e.g. sc.fiji:fiji) that uses imagej.
    :param headless: Whether to start the JVM in headless or gui mode.
    :return: an instance of the net.imagej.ImageJ gateway
    """

    if headless:
        scyjava_config.add_options('-Djava.awt.headless=true')

    if ij_dir_or_version_or_endpoint is None:
        # Use latest release of ImageJ.
        _debug('Using newest ImageJ release')
        scyjava_config.add_endpoints('net.imagej:imagej')

    elif os.path.isdir(ij_dir_or_version_or_endpoint):
        # Assume path to local ImageJ installation.
        path = ij_dir_or_version_or_endpoint
        _debug('Local path to ImageJ installation given: ' + path)
        num_jars = set_ij_env(path)
        print("Added " + str(num_jars + 1) + " JARs to the Java classpath.")
        plugins_dir = str(Path(ij_dir, 'plugins'))
        scyjava_config.add_options('-Dplugins.dir=' + plugins_dir)

    elif ':' in ij_dir_or_version_or_endpoint:
        # Assume endpoint of an artifact.
        endpoint = ij_dir_or_version_or_endpoint
        _debug('Maven coordinate given: ' + endpoint)
        scyjava_config.add_endpoints(endpoint)

    else:
        # Assume version of net.imagej:imagej.
        version = ij_dir_or_version_or_endpoint
        _debug('ImageJ version given: ' + version)
        scyjava_config.add_endpoints('net.imagej:imagej:' + version)

    # Must import imglyb (not scyjava) to spin up the JVM now.
    import imglyb
    from jnius import autoclass

    # Initialize ImageJ.
    ImageJ = autoclass('net.imagej.ImageJ')
    ij = ImageJ()

    # Append some useful utility functions to the ImageJ gateway.

    from scyjava import jclass, isjava, to_java, to_python
    from matplotlib import pyplot

    Dataset                  = autoclass('net.imagej.Dataset')
    RandomAccessibleInterval = autoclass('net.imglib2.RandomAccessibleInterval')

    class ImageJPython:
        def __init__(self, ij):
            self._ij = ij

        def dims(self, image):
            if isinstance(image, numpy.ndarray):
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

        def new_numpy_image(self, image):
            """
            Creates a numpy image (NOT a Java image)
            dimensioned the same as the given image.
            """
            return numpy.zeros(self.dims(image))

        def rai_to_numpy(self, rai):
            result = self.new_numpy_image(rai)
            self._ij.op().run("copy.rai", self.to_java(result), rai)
            return result

        def run_macro(self, macro, args=None):
            if args is None:
                return self._ij.script().run("macro.ijm", macro, True).get()
            else:
                return self._ij.script().run("macro.ijm", macro, True, to_java(args)).get()

        def run_script(self, language, script, args=None):
            script_lang = self._ij.script().getLanguageByName(language)
            if script_lang is None:
                script_lang = self._ij.script().getLanguageByExtension(language)
            if script_lang is None:
                raise ValueError("Unknown script language: " + language)
            exts = script_lang.getExtensions()
            if exts.isEmpty():
                raise ValueError("Script language '" + script_lang.getLanguageName() + "' has no extensions")
            ext = exts.get(0)
            return self._ij.script().run("script." + ext, script, True, to_java(args)).get()

        def to_java(self, data):
            if type(data) == numpy.ndarray:
                return imglyb.to_imglib(data)
            return to_java(data)

        def to_dataset(self, data):
            if self._ij.convert().supports(data, Dataset):
                return self._ij.convert().convert(data, Dataset)
            raise TypeError('Cannot convert to dataset: ' + str(type(data)))

        def from_java(self, data):
            if not isjava(data): return data
            if self._ij.convert().supports(data, Dataset):
                # HACK: Converter exists for ImagePlus -> Dataset, but not ImagePlus -> RAI.
                data = self._ij.convert().convert(data, Dataset)
            if (self._ij.convert().supports(data, RandomAccessibleInterval)):
                rai = self._ij.convert().convert(data, RandomAccessibleInterval)
                return self.rai_to_numpy(rai)
            return to_python(data)

        def show(self, image):
            if image is None:
                raise TypeError('Image must not be None')
            pyplot.imshow(self.from_java(image), interpolation='nearest')
            pyplot.show()

    ij.py = ImageJPython(ij)
    return ij


def imagej_main():
    import sys
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
