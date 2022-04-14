# Developing PyImageJ

This document describes how to do development-related tasks,
if you want to hack on the PyImageJ code itself. If your goal
is only to *use* PyImageJ to call ImageJ and friends from
Python, you do not need to follow any of these instructions.

## Configuring a conda environment for development

```
conda install mamba -n base
mamba env create -f dev-environment.yml
conda activate pyimagej-dev
```

## Running the automated tests

```
./test.sh
```

## Building API documentation

```
cd doc/rtd
make html
```

## Formatting the code

```
black src test
```

## Building distribution bundles

```
python -m build
```
