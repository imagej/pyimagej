import jpype.imports
import scyjava_config

def print_classpath():

    from java.lang import System

    # get classpath and split
    classpath = System.getProperty("java.class.path")
    x = classpath.split(':')

    # print each element
    print('[DEBUG] classpath:')
    for i in x:
        print(i)

    return

def print_endpoints():

    endpoints = scyjava_config.get_endpoints()
    print('[DEBUG] endpoints: {0}'.format(endpoints))

    return