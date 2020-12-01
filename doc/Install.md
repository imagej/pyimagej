# Installation

There are two supported ways to install pyimagej: via
[conda](https://conda.io/) or via
[pip](https://packaging.python.org/guides/tool-recommendations/).
Although both tools are great
[for different reasons](https://www.anaconda.com/blog/understanding-conda-and-pip),
if you have no strong preference then we suggest using conda because it will
manage pyimagej's non-Python dependencies
[OpenJDK](https://en.wikipedia.org/wiki/OpenJDK) (a.k.a. Java) and
[Maven](https://maven.apache.org/). If you use pip, you will need to install
those two things separately.

## Installing via conda

1. Install [Conda](https://conda.io/):
    * On Windows, install Conda using [Chocolatey](https://chocolatey.org): `choco install miniconda3`
    * On macOS, install Conda using [Homebrew](https://brew.sh): `brew cask install miniconda`
    * On Linux, install Conda using its [RPM or Debian package](https://www.anaconda.com/rpm-and-debian-repositories-for-miniconda/), or [with the Miniconda install script](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html).

2. Configure your shell for use with conda:
   ```
   conda init bash
   ```
   Where `bash` is the shell you use.
   Then start a new shell instance.

3. [Activate the conda-forge channel](https://conda-forge.org/docs/user/introduction.html#how-can-i-install-packages-from-conda-forge):
   ```
   conda config --add channels conda-forge
   conda config --set channel_priority strict
   ```

4. Install pyimagej into a new conda environment:
   ```
   conda create -n pyimagej pyimagej openjdk=8
   ```

   This command will install pyimagej with OpenJDK 8. If you would rather use
   OpenJDK 11, you can write `openjdk=11` instead, or even just leave off the
   `openjdk` part altogether to get the latest version of OpenJDK. PyImageJ has
   been tested most thoroughly with Java 8, but it is also known to work with
   Java 11, and likely later Java versions as well.

5. Whenever you want to use pyimagej, activate its environment:
   ```
   conda activate pyimagej
   ```

## Installing via pip

If installing via pip, we recommend using a
[virtualenv](https://virtualenv.pypa.io/) to avoid cluttering up or mangling
your system-wide or user-wide Python environment. Alternately, you can use
conda just for its virtual environment feature (`conda create -n pyimagej
python=3.8; conda activate pyimagej`) and then simply `pip install` everything
into that active environment.

There are several ways to install things via pip, but we will not enumerate
them all here; these instructions will assume you know what you are doing if
you chose this route over conda above.

1. Install [Python 3](https://python.org/). As of this writing, pyimagej has
   been tested with Python 3.6, 3.7, and 3.8. You might have issues with Python
   3.9 on Windows.

2. Install OpenJDK 8 or OpenJDK 11. PyImageJ should work with whichever
   distribution of OpenJDK you prefer; we recommend
   [Zulu JDK+JX 8](https://www.azul.com/downloads/zulu-community/?version=java-8-lts&package=jdk-fx).
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
python -c 'import imagej; ij = imagej.init('2.1.0'); print(ij.getVersion()); ij.dispose()'
```
Should print `2.1.0` on the console.

## Dynamic installation from inside a notebook

### Dynamic installation within Jupyter

It is possible to dynamically install pyimagej from within a Jupyter notebook.

For your first cell, write:
```
import sys
!conda install --yes --prefix {sys.prefix} -c conda-forge pyimagej openjdk=8
```

This approach is useful for [JupyterHub](https://jupyter.org/hub) on the cloud,
e.g. [Binder](https://mybinder.org/), to utilize pyimagej in select notebooks
without advance installation. This reduces time needed to create and launch the
environment, at the expense of a longer startup time the first time a
pyimagej-enabled notebook is run. See [this itkwidgets example
notebook](https://github.com/InsightSoftwareConsortium/itkwidgets/blob/v0.24.2/examples/ImageJImgLib2.ipynb)
for an example.

### Dynamic installation within Google Colab

It is possible to dynamically install pyimagej on
[Google Colab](https://colab.research.google.com/).
See [this thread](https://forum.image.sc/t/pyimagej-on-google-colab/32804)
for guidance. A major advantage of Google Colab is free GPU in the cloud.
