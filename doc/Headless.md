# Using ImageJ in `headless` mode

The original ImageJ was intended to be a GUI-based desktop application with one use, thus it does not natively support true `headless` operation. In order to run ImageJ headlessly, some of ImageJ's core classes are modified via [Javassist](https://imagej.net/develop/javassist) with the [ImageJ patcher](https://github.com/imagej/ij1-patcher/). For more information on how ImageJ achieves `headless` operation please read the [Running Headless](https://imagej.net/learn/headless) page. Note however, even with the current implementation of `headless-imagej2`, not all ImageJ functions are accessible (_e.g._ interacting with the `RoiManager` is not possible with out a frame buffer -- see the [Xvfb](#Xvfb) section of this document).

There are three options to running PyImageJ in headless mode:

## imagej.init(mode='headless')

TODO: ADD INFO

## Xvfb

Xvfb creates a virtual frame buffer for ImageJ's GUI operations without displaying any screen output. On Linux systems that already have a graphical environment installed (_e.g._ GNOME), you only need to install `xvfb`. 

```console
$ sudo apt install xvfb
```

However on fresh Linux servers that do not have any installed environment (_e.g._ Ubuntu Server 20.04.3 LTS), additional X11 related packages will need to be installed for PyImageJ.

```console
sudo apt install libxrender1 libxtst6 libxi6 fonts-dejavu fontconfig
```

After `xvfb` has been installed you can have `xvfb` create the virtual display for you and run a script with:

```console
$ xvfb-run -a python script.py
```

Alternatively you can create the virtual frame buffer manually before you start your PyImageJ session:

```console
$ export DISPLAY=:1
$ Xvfb $DISPLAY -screen 0 1400x900x16 &
```

In either case however, you need to initialize PyImageJ in `interactive` and not `headless` mode so the GUI can be created in the virtual display:

```python
import imagej

ij = imagej.init(mode='interactive')
```
