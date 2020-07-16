# Installation

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

5. Whenever you want to use pyimagej, activate its environment:
    ```
    conda activate pyimagej
    ```

## Installation asides

* The above command installs pyimagej with OpenJDK 8; if you leave off the
  `openjdk=8` it will install OpenJDK 11 by default, which should also work, but
  is less well tested and may have more rough edges.

* It is possible to dynamically install pyimagej from within a Jupyter notebook:
    ```
    import sys
    !conda install --yes --prefix {sys.prefix} -c conda-forge pyimagej openjdk=8
    ```
  This approach is useful for [JupyterHub](https://jupyter.org/hub) on the
  cloud, e.g. [Binder](https://mybinder.org/), to utilize pyimagej in select
  notebooks without advance installation. This reduces time needed to create
  and launch the environment, at the expense of a longer startup time the first
  time a pyimagej-enabled notebook is run. See [this itkwidgets example
  notebook](https://github.com/InsightSoftwareConsortium/itkwidgets/blob/v0.24.2/examples/ImageJImgLib2.ipynb)
  for an example.

* It is possible to dynamically install pyimagej on
  [Google Colab](https://colab.research.google.com/). See
  [this thread](https://forum.image.sc/t/pyimagej-on-google-colab/32804) for
  guidance. A major advantage of Google Colab is free GPU in the cloud.

* If you would prefer to install pyimagej via pip, more legwork is required.
  See [this thread](https://forum.image.sc/t/how-do-i-install-pyimagej/23189/4)
  for hints.
