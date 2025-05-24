# Installation

There are two supported ways to install PyImageJ: via
[conda](https://conda.io/)/[mamba](https://mamba.readthedocs.io/) or via
[pip](https://packaging.python.org/guides/tool-recommendations/).
Both tools are great
[for different reasons](https://www.anaconda.com/blog/understanding-conda-and-pip).

## Installing via conda/mamba

Note: We strongly recommend using
[Mamba](https://mamba.readthedocs.io/en/latest/user_guide/mamba.html) rather
than plain Conda, because Conda is unfortunately terribly slow at configuring
environments.

1. [Install Miniforge3](https://github.com/conda-forge/miniforge#miniforge3).
   **OR:** If you already have `mamba` installed, activate conda-forge:
   ```
   conda config --add channels conda-forge
   conda config --set channel_priority strict
   ```

2. Install PyImageJ into a new environment:
   ```
   mamba create -n pyimagej pyimagej openjdk=11
   ```

   This command will install PyImageJ with OpenJDK 11. PyImageJ requires a
   minimum of OpenJDK 8. PyImageJ has been tested most thoroughly with OpenJDKs
   8 and 11, but it is likely to work with later OpenJDK versions as well.

   *Please note that openjdk=8 from conda-forge is broken on M1 Mac.*
   If you are using an M1 Mac, you should use openjdk=11 or newer.

3. Whenever you want to use PyImageJ, activate its environment:
   ```
   mamba activate pyimagej
   ```

## Installing via pip

If installing via pip, we recommend using a
[virtualenv](https://virtualenv.pypa.io/) to avoid cluttering up or mangling
your system-wide or user-wide Python environment. Alternately, you can use
mamba just for its virtual environment feature (`mamba create -n pyimagej
python=3.9; mamba activate pyimagej`) and then simply `pip install` everything
into that active environment.

There are several ways to install things via pip, but we will not enumerate
them all here; these instructions will assume you know what you are doing if
you chose this route over conda/mamba above.

1. Install [Python 3](https://python.org/). As of this writing,
   PyImageJ has been tested with Python up through 3.13.

2. Install OpenJDK 8 or OpenJDK 11. PyImageJ should work with whichever
   distribution of OpenJDK you prefer; we recommend
   [Zulu JDK+FX 8](https://www.azul.com/downloads/zulu-community/?version=java-8-lts&package=jdk-fx).
   Another option is to install openjdk from your platform's package manager.

3. Install Maven. You can either
   [download it manually](https://maven.apache.org/) or install it via your
   platform's package manager, if available there. The `mvn` command must be
   available on your system path.

4. Install pyimagej via pip:
   ```
   pip install pyimagej
   ```

## Testing your installation

Here's one way to test that it works:
```
python -c 'import imagej; ij = imagej.init("2.14.0"); print(ij.getVersion())'
```
Should print `2.14.0` on the console.

## Dynamic installation within Jupyter

It is possible to dynamically install PyImageJ from within a Jupyter notebook.

For your first cell, write:
```
import sys, os
prefix = sys.prefix.replace("\\", "/") # Handle Windows Paths
%mamba install --yes --prefix {prefix} -c conda-forge pyimagej openjdk=11
jvm_lib_path = [sys.prefix, 'lib', 'jvm']

# platform specific JVM lib path locations
if sys.platform == "win32":
    jvm_lib_path.insert(1, 'Library')
os.environ['JAVA_HOME'] = os.sep.join(jvm_lib_path)
```

This approach is useful for [JupyterHub](https://jupyter.org/hub) on the cloud,
e.g. [Binder](https://mybinder.org/), to utilize PyImageJ in select notebooks
without advance installation. This reduces time needed to create and launch the
environment, at the expense of a longer startup time the first time a
PyImageJ-enabled notebook is run. See [this itkwidgets example
notebook](https://github.com/InsightSoftwareConsortium/itkwidgets/blob/v0.24.2/examples/ImageJImgLib2.ipynb)
for an example.

## Dynamic installation within Google Colab

It is possible to dynamically install PyImageJ on
[Google Colab](https://colab.research.google.com/).
A major advantage of Google Colab is free GPU in the cloud.

Here is an example set of notebook cells to run PyImageJ
on Google Colab with a wrapped local Fiji installation:

1.  Install [condacolab](https://pypi.org/project/condacolab/):
    ```bash
    !pip install -q condacolab
    import condacolab
    condacolab.install()
    ```

2.  Verify that the installation is functional:
    ```python
    import condacolab
    condacolab.check()
    ```

3.  Install PyImageJ:
    ```bash
    !mamba install pyimagej openjdk=11
    ```
    You can also install other deps here as well (scikit-image, opencv, etc).

4.  Download and install Fiji, and optionally custom plugins as well:
    ```bash
    !wget https://downloads.imagej.net/fiji/latest/fiji-linux64.zip > /dev/null && unzip fiji-linux64.zip > /dev/null
    !rm fiji-linux64.zip
    !wget  https://imagej.net/ij/plugins/download/Filter_Rank.class > /dev/null
    !mv Filter_Rank.class Fiji.app/plugins
    ```

5.  Set `JAVA_HOME`:
    ```python
    import os
    os.environ['JAVA_HOME']='/usr/local'
    ```
    We need to do this so that the openjdk installed by mamba gets used,
    since a conda env is not actually active in this scenario.

6.  Start PyImageJ wrapping the local Fiji:
    ```python
    import imagej
    ij = imagej.init("/content/Fiji.app")
    print(ij.getVersion())
    ```

7.  Start running plugins, even custom plugins:
    ```python
    imp = ij.IJ.openImage("http://imagej.net/images/blobs.gif")
    ij.py.run_plugin("Filter Rank", {"window": 3, "randomise": True}, imp=imp)
    ij.IJ.resetMinAndMax(imp)
    ij.py.run_plugin("Enhance Contrast", {"saturated": 0.35}, imp=imp)
    ```
## Install pyimagej in Docker
We leverage [Micromamba-docker](https://github.com/mamba-org/micromamba-docker) since `conda activate` will not work. Note that running Python scripts during build is extremely slow.
```dockerfile
# Micromamba-docker @ https://github.com/mamba-org/micromamba-docker
FROM mambaorg/micromamba:1.0.0

# Retrieve dependencies
USER root
RUN apt-get update
RUN apt-get install -y wget unzip > /dev/null && rm -rf /var/lib/apt/lists/* > /dev/null
RUN micromamba install -y -n base -c conda-forge \
        python=3.9\
        pyimagej  \
        openjdk=11 && \
    micromamba clean --all --yes
ENV JAVA_HOME="/usr/local"
# Set MAMVA_DOCKERFILE_ACTIVATE (otherwise python will not be found)
ARG MAMBA_DOCKERFILE_ACTIVATE=1  
# Retrieve ImageJ and source code
RUN wget https://downloads.imagej.net/fiji/latest/fiji-linux64.zip &> /dev/null
RUN unzip fiji-linux64.zip > /dev/null
RUN rm fiji-linux64.zip
# test: note that "Filter Rank" is not added yet, below is just an example.
RUN python -c "import imagej; \
    ij = imagej.init('/tmp/Fiji.app', mode='headless'); \
    print(ij.getVersion()); \
    imp = ij.IJ.openImage('http://imagej.net/images/blobs.gif'); \
    ij.py.run_plugin('Filter Rank', {'window': 3, 'randomise': True}, imp=imp); \
    ij.IJ.resetMinAndMax(imp); \
    ij.py.run_plugin('Enhance Contrast', {'saturated': 0.35}, imp=imp);"
```
