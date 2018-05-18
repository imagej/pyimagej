"""
wrapper for imagej and python integration using Pyjnius

"""

__version__ = '0.1.0'
__author__ = 'Yang Liu'

import subprocess
import os
import sys
import re
import jnius_config


def _debug(message):
    """
    print debug message

    :param message: Debug message to be printed
    :return: None
    """
    if not __debug__:
        print(message)


def setenv(k, v):
    """
    set up an general environment variable

    :param k: Environment name
    :param v: Environment value
    :return: None
    """

    os.environ[k] = v


def getenv(k):
    """
    print the environment variable to console

    :param k: Environment name
    :return: None
    """

    print(os.getenv(k))


def set_conda_env(conda_env_path):
    """
    Setup Conda environment path

    :param conda_env_path: Environment name
    :return: None
    """

    setenv('CONDA_PREFIX', conda_env_path)


def set_java_env(java_home_path):
    """
    Setup Java environment path

    :param java_home_path: System path for java
    :return: None
    """

    setenv('JAVA_HOME', java_home_path)


def set_pyjnius_env(pyjnius_dir):
    """
    set up an pyjnius environment path

    :param pyjnius_dir: System path for pyjnius
    :return: None
    """

    if pyjnius_dir is None:
        print("pyjnius directory is not correct")
        return
    else:
        setenv('PYJNIUS_JAR', pyjnius_dir)


def set_imglyb_env(imglyb_dir):
    """
    set up the variable path for imglyb

    :param imglyb_dir: local path to the imglyb jar
    :return: None
    """

    if imglyb_dir is None:
        print("classpath entered is not correct")
        return
    else:
        setenv('IMGLYB_JAR', imglyb_dir)
        return


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
            if f.endswith('.jar') and \
                    'imagej-legacy' not in f and \
                    'ij1-patcher' not in f and \
                    'ij-1' not in f and \
                    'legacy-imglib1' not in f:
                path = root + '/' + f
                jars.append(path)
                _debug('Added ' + path)
    return jars


def set_ij_env(ij_dir, imglyb_path):
    """
    Create a list of required jars and add to imglyb search path

    :param ij_dir: System path for Fiji.app
    :param imglyb_path: System path for imglyb
    :return: num_jar(int): number of jars added
    """
    jars = []
    # search jars directory
    jars.extend(search_for_jars(ij_dir, '/jars'))
    # search plugins directory
    jars.extend(search_for_jars(ij_dir, '/plugins'))
    # add to classpath
    num_jars = len(jars)
    classpath = ":".join(jars) + ":" + imglyb_path
    set_imglyb_env(classpath)
    return num_jars


def verify_java_env():
    """
    Verify System variable has JAVA_HOME and it is valid

    :return: None
    """

    if os.getenv('JAVA_HOME') is None:
        print('Java Environment is not set correctly, \
                please set Java Environment by using set_java_env(your_local_path_to_java')
        return
    else:
        java_home = os.getenv('JAVA_HOME')
        if os.path.isfile(java_home + '/bin/java'):
            print('Java environment: ' + os.getenv('JAVA_HOME'))
            return
        else:
            print('Java Environment is not set correctly, \
                            please set Java Environment by using set_java_env(your_local_path_to_java')
            return


def verify_conda_env():
    """
    make sure the conda env is correct

    :return: None
    """

    conda_env = os.getenv('CONDA_PREFIX')
    if conda_env is None:
        print('Conda environment is not set, \
                please manually set the conda enviroment')
    else:
        try:
            subprocess.check_output([conda_env + '/bin/conda', '--version'])
        except OSError:
            print('Conda Environment is not set correctly,\
                    please manually set the conda enviroment')
            return None
        print('Conda environment: ' + conda_env)
        return conda_env


def init(ij_dir):
    """
    quietly set up the whole environment

    :param ij_dir: System path for Fiji.app
    :return: an instance of the net.imagej.ImageJ gateway
    """

    jnius_config.add_options('-Djava.awt.headless=true')
    imglyb_path = configure_path()
    # ImageJ
    if imglyb_path is not None:
        num_jars = set_ij_env(ij_dir, imglyb_path)
    else:
        return
    print("Added " + str(num_jars + 1) + " JARs to the Java classpath.")
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


def error_message(error):
    """
    print error message

    :param error: The name of the file that can not be found
    :return: None
    """
    print (error + " can not be found, it might not be correctly installed.")


def jar_present(path, basedir, test_path, target):
    """
    search for the target jar

    :param path: inherited target path, if already found then skip
    :param test_path: the path need to be checked
    :param target: the name of the target
    :return: target path
    """
    if path is None and os.path.isdir(test_path) and os.listdir(test_path) is not None:
        _debug('Scanning directory: ' + test_path)
        for f in os.listdir(test_path):
            if 'java' is not target:
                if ".jar" in f:
                    result_path = test_path + f
                    _debug('Found' + target + ' at: ' + result_path)
                    return result_path
            else:
                if "java" in f:
                    result_path = basedir
                    _debug('Found' + target + ' at: ' + result_path)
                    return result_path

    else:
        return path


def conda_path_check(p, checked, imglyb_path, pyjnius_path, java_path):
    """
    search for imglyb, pyjnius and java path if this is a conda environment

    :param p: base path to check
    :param checked: list of paths have already been checked
    :param imglyb_path: if already found, path to imglyb, if not found, None
    :param pyjnius_path: if already found, path to pyjnius, if not found, None
    :param java_path: if already found, path to java, if not found, None
    :return: imglyb_path: path to imglyb if found, otherwise None
    :return: pyjnius_path: path to pyjnius if found, otherwise None
    :return: java_path: path to java if found, otherwise None
    """
    split_list = p.split("/")
    index_conda = 0
    index_env = 0

    for level in split_list:
        index_env += 1
        if "conda" in level:
            index_conda = index_env-1
        if level == "envs":
            break
    
    if index_env == len(split_list):
        index = index_conda
    else:
        index = index_env

    basedir = "/".join(split_list[0:index+1])
    if basedir in checked:
        return imglyb_path, pyjnius_path, java_path

    test_path_imglyb = basedir + "/share/imglyb/"
    test_path_pyjnius = basedir + "/share/pyjnius/"
    test_path_java = basedir + "/bin"

    imglyb_path = jar_present(imglyb_path, basedir, test_path_imglyb, 'imglyb')
    pyjnius_path = jar_present(pyjnius_path, basedir, test_path_pyjnius, 'pyjnius')
    java_path = jar_present(java_path, basedir, test_path_java, 'java')
    checked.append(basedir)
    return imglyb_path, pyjnius_path, java_path


def pypi_path_check(p, checked, imglyb_path, pyjnius_path):
    """
    check path if python is installed through pip

    :param p: current checking path
    :param checked: list of checked path
    :param imglyb_path: if already found, path to imglyb, if not found, None
    :param pyjnius_path: if already found, path to pyjnius, if not found, None
    :return:
    """

    split_list = p.split("/")
    index = 0
    for level in split_list:
        index += 1
        if level == "site_packages" or level == "dist-packages":
            break

    basedir = "/".join(split_list[0:index + 1])
    if basedir in checked:
        return None, None

    test_path_imglyb = basedir + "/imglyb/"
    test_path_pyjnius = basedir + "/pyjnius/"

    if imglyb_path is None and os.path.isdir(test_path_imglyb):
        for f in os.listdir(test_path_imglyb):
            if ".jar" in f:
                imglyb_path = test_path_imglyb + f

    if pyjnius_path is None and os.path.isdir(test_path_pyjnius):
        for f in os.listdir(test_path_pyjnius):
            if ".jar" in f:
                pyjnius_path = test_path_pyjnius + f

    checked.append(basedir)
    return imglyb_path, pyjnius_path


def configure_path():
    """
    find the path to imglyb, pyjnius and java by searching though python syspath

    :return: imglyb_path: the path to imglyb if found
    """
    paths = sys.path
   
    imglyb_path = None
    pyjnius_path = None
    java_path = None

    checked = []
    index = 0

    while index < len(paths) and (imglyb_path is None or pyjnius_path is None or java_path is None):
        p = paths[index]
        _debug('Checking path: ' + p)
        if "conda" in p:
            imglyb_path, pyjnius_path, java_path = conda_path_check(p, checked, imglyb_path, pyjnius_path, java_path)
        elif "site-packages" or "dist-packages" in p:
            imglyb_path, pyjnius_path = pypi_path_check(p, checked, imglyb_path, pyjnius_path)
        index += 1
    
    if pyjnius_path is None:
        error_message("pyjnius")
    else:
        set_pyjnius_env(pyjnius_path)

    if java_path is None:
        java_path = java_check()
        if java_path is not None:
            set_java_env(java_path)
        else:
            error_message("Java")
    else:
        set_java_env(java_path)

    if imglyb_path is None:
        error_message("imglyb")
    else:
        return imglyb_path
    return


def os_check():
    """
    check the os of current pc

    :return: os type
    """

    return sys.platform


def java_check():
    """
    Try to find java in using "which" and "whereis" command according to os type

    :return: each: the path to java
    """
    system_name = os_check()
    if "linux" in system_name:
        path_java_home_l = subprocess.check_output(["echo", "$JAVA_HOME"], shell=True)
        if path_java_home_l is None or path_java_home_l == "\n":
            path_linux = subprocess.check_output(["which", "java"])
            if path_linux is None:
                java_list = subprocess.check_output(["whereis", "java"]).split(" ")
                java_list = java_list[1:len(java_list) - 1]
                for each in java_list:
                    try:
                        output = subprocess.check_output([each, "-version"], stderr=subprocess.STDOUT)
                        version = re.search('"(.+?)"', output).group(1)
                        if "9" in version[0:3]:
                            pass
                        else:
                            return each
                    except OSError:
                        pass

        return None

    elif "darwin" in system_name:
        path_java_home_d = subprocess.check_output(["echo", "$JAVA_HOME"], shell=True)
        if path_java_home_d is None or path_java_home_d == "\n":
            path_darwin = subprocess.check_output(["which", "java"])
            if path_darwin is None:
                java_list = subprocess.check_output(["whereis", "java"]).split(" ")
                java_list = java_list[1:len(java_list) - 1]
                for each in java_list:
                    try:
                        output = subprocess.check_output([each, "-version"], stderr=subprocess.STDOUT)
                        version = re.search('"(.+?)"', output).group(1)
                        if "9" in version[0:3]:
                            pass
                        else:
                            return each
                    except OSError:
                        pass
                
        return None
    elif "win32" in system_name:
        print("please set the java enviroment manully by call set_java_env() command")
        return None
