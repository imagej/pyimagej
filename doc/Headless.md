# Using ImageJ in `headless` mode

The original ImageJ was intended to be a GUI-based desktop application with one user, thus it does not natively support true `headless` operation. In order to run ImageJ headlessly, some of ImageJ's core classes are modified via [Javassist](https://imagej.net/develop/javassist) with the [ImageJ patcher](https://github.com/imagej/ij1-patcher/). For more information on how ImageJ achieves `headless` operation please read the [Running Headless](https://imagej.net/learn/headless) page. Note however, even with the current implementation of `headless-imagej2`, not all ImageJ functions are accessible (_e.g._ interacting with the `RoiManager` is not possible with out a frame buffer -- see the [Xvfb](#Xvfb) section of this document).

There are three options to running PyImageJ in headless mode:

## imagej.init(mode='headless')

PyImageJ `headless` mode can be selected during the initialization step. Under the hood the `headless` mode flag is handled by scjava and JPype and initializes the JVM with `-Djava.awt.headless=true`.

```python
import imagej

ij = imagej.init(mode='headless')
```

**Warning:** ImageJ does not completely support headless operation. Various ImageJ plugins and features will not work headlessly (_e.g._ the WindowManager and RoiMAnager do not support headaless environments, see [Xvfb](#Xvfb))

## Xvfb
 
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

## Examples

Below we have two examples on how to run PyImageJ headlessly (1) using `imagej.init(mode='interactive')` and (2) using Xvfb.

### **(2) headless xvfb example**

In addition to Xvfb, you will also need to have scikit-image installed in your environment to run the `doc/examples/headless_blob_detection.py` example. The `headless_blob_detection.py` script is the headless version of the `doc/examples/blob_detection.py` example (please run `blob_detection.py` to view the scikit-image blob detection output). 

The headless example opens the `test_image.tif` sample image, detects the blobs via scikit-image's Laplacian of Gaussian algorithm, adds the blob detections to the ImageJ `RoiManager`, measures the ROIs and returns a panda's dataframe of the measurement results. To run the example, run the following command to create the virtual frame buffer and run PyImageJ:

```console
$ xvfb-run -a python headless_blob_detection.py
```

The script should print the results pandas dataframe (the data from ImageJ's `ResultsTable`) with 187 detections.

```python
log4j:WARN No appenders could be found for logger (org.bushe.swing.event.EventService).
log4j:WARN Please initialize the log4j system properly.
log4j:WARN See http://logging.apache.org/log4j/1.2/faq.html#noconfig for more info.
ImageJ version: 2.3.0/1.53f
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