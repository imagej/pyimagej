import os
import logging
import jgo
import jpype
import jpype.imports
import scyjava_config

from pathlib import Path

# setup logger
_logger = logging.getLogger(__name__)    

def start_JVM():

    # get endpoints and repos
    endpoints = scyjava_config.get_endpoints()
    print('[DEBUG] endpoints from scyjava:\n{0}'.format(endpoints))
    repositories = scyjava_config.get_repositories()
    print('[DEBUG] repos from scyjava:\n{0}'.format(repositories))

    # add more endpoints and repos
    if len(endpoints) > 0:
        endpoints = endpoints[:1] + sorted(endpoints[1:])
        _logger.debug('Using endpoints %s', endpoints)
        _, workspace = jgo.resolve_dependencies(
            '+'.join(endpoints),
            m2_repo=scyjava_config.get_m2_repo(),
            cache_dir=scyjava_config.get_cache_dir(),
            manage_dependencies=scyjava_config.get_manage_deps(),
            repositories=repositories,
            verbose=scyjava_config.get_verbose()
        )
        jpype.addClassPath(os.path.join(workspace, '*'))

    # endponts prior to jvm initialization
    print('[DEBUG] endpoints before jvm:\n{0}'.format(endpoints))
    print('[DEBUG] repos before jvm:\n{0}'.format(repositories))
    jpype.startJVM()

    if jpype.isJVMStarted() == True:
        print('[DEBUG] JVM status: Started')

    return

def print_classpath():

    from java.lang import System

    print('[DEBUG] classpath:\n{0}'.format(System.getProperty("java.class.path")))

    return

def init_ij():

    global  ij

    ImageJ = jpype.JClass('net.imagej.ImageJ')
    ij = ImageJ()

    return

start_JVM()