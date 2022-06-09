# Using PyImageJ without a screen

It is an increasingly common scenario to want to do image processing on a cloud
computing node (e.g. running notebooks on [Binder](https://mybinder.org/) or
[Google Colab](https://colab.research.google.com)). Unfortunately, the
original ImageJ was only designed to be a GUI-based desktop application, so it
does not natively support true
[headless](https://en.wikipedia.org/wiki/Headless_computer) operation, i.e.
without a display attached.

## Using ImageJ in headless mode

The [ImageJ2](https://imagej.net/software/imagej2) project
supports headless operation for all its functions, due to its careful
[separation of concerns](https://imagej.net/develop/architecture#modularity),
and ImageJ2 includes a
[backwards compatibility layer](https://imagej.net/libs/imagej-legacy)
that supports use some original ImageJ functionality while headless;
the original ImageJ's core classes are
[modified at runtime via Javassist](https://github.com/imagej/ij1-patcher).

For more information about running ImageJ and/or ImageJ2 in headless mode,
please read the [Running Headless](https://imagej.net/learn/headless) and
[Scripting Headless](https://imagej.net/scripting/headless) pages of the
ImageJ wiki.

***Please note:*** Not all original ImageJ functions are accessible while
headless: e.g., many methods of `RoiManager` and `WindowManager` do not work
without a graphical environment. To work around this limitation, you can use
[Xvfb](#using-pyimagej-with-xvfb) to run ImageJ inside a "virtual" graphical environment without a
physical screen present.

### Starting PyImageJ in headless mode

When you initialize PyImageJ with no arguments,
it runs in headless mode by default:

```python
import imagej
ij = imagej.init()
```

For clarity, you can explicitly specify headless mode
by passing the `mode='headless'` setting:

```python
ij = imagej.init(mode='headless')
```

Under the hood, the headless mode flag initializes the
Java Virtual Machine with `-Djava.awt.headless=true`.

For more about PyImageJ initialization, see the
[Initialization](Initialization) guide.

### Troubleshooting

See the
[Known Limitations section of the Troubleshooting guide](Troubleshooting.md#known-limitations)
for some further details about what does and does not work headless, and
things to try when having difficulty with ImageJ's behavior in headless mode.


## Using PyImageJ with Xvfb

 Workflows that require headless operation but also need to interact with ImageJ elements that are tied to the GUI, can be achieved with virtual displays. Using Xvfb we can create a virtual frame buffer for ImageJ's GUI elemnts without displaying any screen output. On Linux systems that already have a graphical environment installed (_e.g._ GNOME), you only need to install `xvfb`.

```console
$ sudo apt install xvfb
```

However on fresh Linux servers that do not have any installed environment (_e.g._ Ubuntu Server 20.04.3 LTS), additional X11 related packages will need to be installed for PyImageJ.

```console
$ sudo apt install libxrender1 libxtst6 libxi6 fonts-dejavu fontconfig
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

### **Headless Xvfb example**

Here we have an example on how to run PyImageJ headlessly using `imagej.init(mode='interactive')` and Xvfb. In addition to Xvfb, you will also need to have scikit-image installed in your environment to run the `doc/examples/blob_detection_xvfb.py` example. The `blob_detection_xvfb.py` script is the headless version of the `doc/examples/blob_detection_interactive.py` example (please run `blob_detection_interactive.py` to view the scikit-image blob detection output).

The headless example opens the `test_image.tif` sample image, detects the blobs via scikit-image's Laplacian of Gaussian algorithm, adds the blob detections to the ImageJ `RoiManager`, measures the ROIs and returns a panda's dataframe of the measurement results. To run the example, run the following command to create the virtual frame buffer and run PyImageJ:

```console
$ xvfb-run -a python blob_detection_xvfb.py
```

The script should print the results pandas dataframe (the data from ImageJ's `ResultsTable`) with 187 detections.

```python
log4j:WARN No appenders could be found for logger (org.bushe.swing.event.EventService).
log4j:WARN Please initialize the log4j system properly.
log4j:WARN See http://logging.apache.org/log4j/1.2/faq.html#noconfig for more info.
ImageJ2 version: 2.5.0/1.53r
Output:
         Area         Mean     Min     Max
0    1.267500  3477.416667  2219.0  5312.0
1    0.422500  2075.500000  1735.0  2529.0
2    0.422500  1957.750000  1411.0  2640.0
3    0.422500  1366.500000  1012.0  1913.0
4    0.422500  2358.500000  2100.0  2531.0
..        ...          ...     ...     ...
182  0.422500  1205.750000  1124.0  1355.0
183  7.288125  1362.840580   703.0  2551.0
184  0.422500   920.500000   830.0  1110.0
185  0.422500  1345.250000  1260.0  1432.0
186  0.422500  1097.250000   960.0  1207.0

[187 rows x 4 columns]
```
