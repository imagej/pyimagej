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
