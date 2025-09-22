# Developing PyImageJ

This document describes how to do development-related tasks,
if you want to hack on the PyImageJ code itself. If your goal
is only to *use* PyImageJ to call ImageJ and friends from
Python, you do not need to follow any of these instructions.

## Running the automated tests

```
make test
```

## Building the reference documentation

```
make html
```

Results are generated to `doc/_build/html`.
Production documentation is available online at
[https://pyimagej.readthedocs.io/](https://pyimagej.readthedocs.io/).

## Formatting the code

```
make lint
```

## Building distribution bundles

```
make dist
```
