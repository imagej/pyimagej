# Developing PyImageJ

This document describes how to do development-related tasks,
if you want to hack on the PyImageJ code itself. If your goal
is only to *use* PyImageJ to call ImageJ and friends from
Python, you do not need to follow any of these instructions.

## Configuring a conda environment for development

Install [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge).
Then:

```
mamba env create -f dev-environment.yml
mamba activate pyimagej-dev
```

## Running the automated tests

```
./test.sh
```

## Building the reference documentation

```
cd doc/
make html
```

Results are generated to `doc/_build/html`.
Production documentation is available online at
[https://pyimagej.readthedocs.io/](https://pyimagej.readthedocs.io/).

## Formatting the code

```
black src test
```

## Building distribution bundles

```
python -m build
```
