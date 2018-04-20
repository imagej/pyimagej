"""
wrapper for imagej and python integration using Pyjnius

"""

__version__ = '0.1.0'
__author__ = 'Yang Liu'

import subprocess
import os
import sys
import re


def setenv(k, v):
    """set up an general environment variable

    Args:
        k(string): Environment name
        v(string): Environment value
    """

    os.environ[k] = v


def getenv(k):
    """print the enviroment 

    Args:
        k(string): Enviroment name
    """
    print(os.getenv(k))


def set_conda_env(conda_env_path):
    """set up an conda environment

    Args:
        conda_env_path(string): System path for conda
    """

    setenv('CONDA_PREFIX', conda_env_path)


def set_java_env(java_home_path):
    """set up an Java environment

    Args:
        java_home_path(string): System path for java
    """

    setenv('JAVA_HOME', java_home_path)


def set_pyjnius_env(pyjnius_dir):
    """set up an pyjnius environment

    Args:
        pyjnius_dir(string): System path for conda

    return: None if pyjnius_dir is not set
    """

    if pyjnius_dir is None:
        print("pyjnius directory is not correct")
        return
    else:
        setenv('PYJNIUS_JAR', pyjnius_dir)


def set_ij_env(ij_dir, imglyb_path):
    """make a list of all the required jar file

    Args:
        ij_dir(string): System path for Fiji.app

    return:
        classpath(string): list of required jars
        num_jar(int): number of jars added
    """

    jars = []
    for root, dirs, files in os.walk(ij_dir + '/jars'):
        for each_file in files:
            if each_file.endswith('.jar') and \
                    'imagej-legacy' not in each_file and \
                    'ij1-patcher' not in each_file and \
                    'ij-1' not in each_file:
                jars.append(root + '/' + each_file)
    num_jars = len(jars)
    classpath = ":".join(jars) + ":" + imglyb_path
    set_imglyb_env(classpath)
    return num_jars


def set_imglyb_env(imglyb_dir):
    """set up the variable path for imglyb

    Args:
        imglyb_dir(string): local path to the imglyb jar
    """

    if imglyb_dir is None:
        print("classpath entered is not correct")
        return
    else:
        setenv('IMGLYB_JAR', imglyb_dir)
        return


def verify_java_env():
    """make sure the java env is correct

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
    """make sure the conda env is correct

    return: conda_env(string): if correct, return conda environment variable
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


def quiet_init(ij_dir):
    """quietly setup the whole environment and run checks

    Args: ij_dir(String): System path for Fiji.app

    """
    imglyb_path = configure_path()
    # ImageJ
    if(imglyb_path is not None):
        num_jars = set_ij_env(ij_dir, imglyb_path)
    else:
        return
    print("Added " + str(num_jars + 1) + " JARs to the Java classpath.")


def help():
    """print the instruction for using imagej module

    """

    print(("Please set the environment variables first:\n" 
           "Fiji.app:   ij_dir = 'your local fiji.app path'\n"
           "Then call quiet_init(ij_dir)"))


def error_message(error):
    print (error + " can not be found, it might not be correctly installed.")


def conda_path_check(p, checked, imglyb_path, pyjnius_path, java_path):
    split_list = p.split("/")
    index_conda = 0
    index_env = 0

    for level in split_list:
        index_env += 1
        if "conda" in level:
            index_conda = index_env
        if level == "envs":
            break
    
    if index_env == len(split_list):
        index = index_conda
    else:
        index = index_env

    basedir = "/".join(split_list[0:index+1])
    if basedir in checked:
        return None, None, None

    test_path_imglyb = basedir + "/share/imglyb/"
    test_path_pyjnius = basedir + "/share/pyjnius/"
    test_path_java = basedir + "/bin"

    if imglyb_path is None and os.path.isdir(test_path_imglyb) and os.listdir(test_path_imglyb) is not None:
        for f in os.listdir(test_path_imglyb):
            if ".jar" in f:
                imglyb_path = test_path_imglyb + f
                break

    if pyjnius_path is None and os.path.isdir(test_path_pyjnius) and os.listdir(test_path_pyjnius) is not None:
        for f in os.listdir(test_path_pyjnius):
            if ".jar" in f:
                pyjnius_path = test_path_pyjnius + f
                break

    if java_path is None and os.path.isdir(test_path_java) and os.listdir(test_path_java) is not None:
        for f in os.listdir(test_path_java):
            if "java" == f:
                java_path = basedir
                break

    checked.append(basedir)
    return imglyb_path, pyjnius_path, java_path


def pypi_path_check(p, checked, imglyb_path, pyjnius_path):
    """
    :param p: current checking path
    :type checked: list
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
    paths = sys.path
   
    imglyb_path = None
    pyjnius_path = None
    java_path = None

    checked = []
    index = 0

    while index < len(paths) and (imglyb_path is None or pyjnius_path is None or java_path is None):
        p = paths[index]
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
    return sys.platform

def java_check():
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
