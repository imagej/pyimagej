"""
wrapper for imagej and python integration using ImgLyb

"""

__version__ = '0.3.1'
__author__ = 'Yang Liu & Curtis Rueden'

import os
import jnius_config
from pathlib import Path


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
    num_jars = len(jars)
    jnius_config.add_classpath(os.pathsep.join(jars))
    return num_jars


def init(ij_dir, headless=True):
    """
    quietly set up the whole environment

    :param ij_dir: System path for Fiji.app
    :return: an instance of the net.imagej.ImageJ gateway
    """
    
    if headless:
        jnius_config.add_options('-Djava.awt.headless=true')
    
    plugins_dir = str(Path(ij_dir, 'plugins'))
    jnius_config.add_options('-Dplugins.dir=' + plugins_dir)
    
    num_jars = set_ij_env(ij_dir)
    print("Added " + str(num_jars + 1) + " JARs to the Java classpath.")
    
    # It is necessary to import imglyb before jnius because it sets options for the JVM and jnius starts up the JVM
    import imglyb
    from jnius import autoclass
    ImageJ = autoclass('net.imagej.ImageJ')
    return ImageJ()


def help():
    """
    print the instruction for using imagej module

    :return:
    """

    print(("Please set the environment variables first:\n" 
           "Fiji.app:   ij_dir = 'your local fiji.app path'\n"
           "Then call init(ij_dir)"))
