import jpype.imports
import scyjava


def print_classpath():
    from java.lang import System

    # get classpath and split
    classpath = System.getProperty('java.class.path')
    x = classpath.split(':')

    # print each element
    print("[DEBUG] classpath:")
    for i in x:
        print(i)


def print_endpoints():
    endpoints = scyjava.config.get_endpoints()
    print("[DEBUG] endpoints: {0}".format(endpoints))


def print_ij_version(ij):
    print("[DEBUG] ij version: {0}".format(ij.getVersion()))


def print_obj_dir(object):
    obj_dir = dir(object)

    print("[DEBUG] object name: {0}\n[DEBUG] dir:".format(str(object)))
    for x in obj_dir:
        print(x)


def print_obj_type(object):
    print("[DEBUG] object name: {0}\n[DEBUG] type: {1}".format(str(object), type(object)))
