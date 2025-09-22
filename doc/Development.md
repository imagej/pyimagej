# Developing PyImageJ

This document describes how to do development-related tasks,
if you want to hack on the PyImageJ code itself. If your goal
is only to *use* PyImageJ to call ImageJ and friends from
Python, you do not need to follow any of these instructions.

## Testing local changes

This project is designed to work with [`uv`
for package management](https://docs.astral.sh/uv/), which automatically manages cached virtual environments. `uv` is invoked as a wrapper around processes, reusing the cached libraries as needed.

If you have local `pyimagej` changes you wanted to test, you can simply start a python interpreter from the project root:

```bash
uv run python
```

All of the following `make` commands implicitly use `uv` in their tasks.

### Local changes in downstream projects

If you are testing a project that *uses* `pyimagej` and need to see how changes in `pyimagej` impact your code, you have two options. Either way, you will need to install `pyimagej` in [editable mode](https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs) into the virtual environment of  your choice:

1. A `uv`-managed virual environment, e.g. `uv venv`
1. A [mamba](https://mamba.readthedocs.io/en/latest/user_guide/mamba.html)-managed environment

## Building the reference documentation

The documentation has its own `Makefile` in the `/docs` directory. From there:

```
make html
```

Alternatively, from the project root:

```
make docs
```

Results are generated to `doc/_build/html`.
Production documentation is available online at
[https://pyimagej.readthedocs.io/](https://pyimagej.readthedocs.io/).

## Running the automated tests

```
make test
```

## Formatting the code

```
make lint
```

## Building distribution bundles

```
make dist
```
