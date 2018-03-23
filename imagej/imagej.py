"""
wrapper for imagej and python integration using pyjnius

"""

__version__ = '0.1.0'
__author__ = 'Yang Liu'


import subprocess
import os


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


def set_pyjnius_env(conda_env):
    """set up an pyjnius environment

    Args:
        conda_env(string): System path for conda

    return: None if conda_env is not set
    """

    if conda_env is None:
        return
    elif os.getenv('PYJNIUS_JAR') is None:
        pyjnius_dir = (conda_env + '/share/pyjnius/')
        setenv('PYJNIUS_JAR', pyjnius_dir + os.listdir(pyjnius_dir)[0])
    else:
        print('PYJNIUS_JAR: ' + os.getenv('PYJNIUS_JAR'))


def set_ij_env(ij_dir):
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
    classpath = ":".join(jars)
    num_jars = len(jars)
    return classpath, num_jars


def set_imglyb_env(conda_env, classpath):
    """set up the variable path for imglyb

    Args:
        conda_env(string): System path for conda
        classpath(string): all the require jar files
    """

    if conda_env is None:
        pass
    else:
        imglyb_dir = conda_env + '/share/imglyb/'
        imglyb_jar = imglyb_dir + os.listdir(imglyb_dir)[0]
        classpath += ':' + imglyb_jar
        setenv('IMGLYB_JAR', classpath)
        return


def verify_java_env():
    """make sure the java env is correct

    """
    
    if os.getenv('JAVA_HOME') is None:
        print('Java Environment is not set correctly, \
                please set Java Environment by execute the top block')
        return
    else:
        java_home = os.getenv('JAVA_HOME')
        if os.path.isfile(java_home + '/bin/java'):
            print('Java environment: ' + os.getenv('JAVA_HOME'))
            return
        else:
            print('Java Environment is not set correctly, \
                            please set Java Environment by execute the top block')
            return

#TODO: make this work with pypi and more
def verify_conda_env():
    """make sure the conda env is correct

    return: conda_env(string): if correct, return conda environment variable
    """

    conda_env = os.getenv('CONDA_PREFIX')
    if conda_env is None:
        print('Conda environment is not set, \
                please execute the top block')
    else:
        try:
            subprocess.check_output([conda_env + '/bin/conda', '--version'])
        except OSError:
            print('Conda Environment is not set correctly,\
                    please set Conda Environment by execute the top block')
            return None
        print('Conda environment: ' + conda_env)
        return conda_env


def quiet_init(ij_dir):
    """quietly setup the whole environment and run checks

    Args: ij_dir(String): System path for Fiji.app

    """

    conda_env = verify_conda_env()
    set_pyjnius_env(conda_env)
    verify_java_env()

    # ImageJ
    classpath, num_jars = set_ij_env(ij_dir)

    # ImgLyb
    set_imglyb_env(conda_env, classpath)
    print("Added " + str(num_jars + 1) + " JARs to the Java classpath.")


def help():
    """print the instruction for using imagej module

    """

    print(("Please set the enviroment variables first:\n"
        "1. conda:      set_conda_env('your local conda path')\n"
        "2. Java:       set_java_env('your local java path')\n"
        "3. Fiji.app:   ij_dir = 'your local fiji.app path'\n" 
    "Then call quiet_init(ij_dir)"))


