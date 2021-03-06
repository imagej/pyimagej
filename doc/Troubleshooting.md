# Common Errors

## I ran a plugin and see an updated image, but the numpy array and dataset are unchanged.

This bug can occur in certain circumstances when using ImageJ1 plugins which
update a corresponding `ImagePlus`. It can be worked around by calling:

```python
imp = ij.py.WindowManager.getCurrentImage()
ij.py.synchronize_ij1_to_ij2(imp)
```

## The GUI has issues on macOS

The default UI may not work on macOS. You can try using ImageJ's
Swing-UI-based GUI instead by initializing with:

```python
import imagej
ij = imagej.init(..., headless=False)
ij.ui().showUI("swing")
```

Replacing `...` with one of the usual possibilities
(see [Initialization.md](Initialization.md)).

See [this thread](https://github.com/imagej/pyimagej/issues/23)
for additional information and updates.

## ImageJ1 classes not found

If you try to load an ImageJ1 class (with package prefix `ij`), and get a
`JavaException: Class not found` error, this is because ImageJ was initialized
without ImageJ1. See [Initialization.md](Initialization.md).

## Not enough memory

You can increase the memory available to the JVM before starting ImageJ.
See [Initialization.md](Initialization.md).

## log4j:WARN 

PyImageJ does not currently ship a log4j implementation, which results in an
obnoxious warning at startup:

```
log4j:WARN No appenders could be found for logger (org.bushe.swing.event.EventService).
log4j:WARN Please initialize the log4j system properly.
log4j:WARN See http://logging.apache.org/log4j/1.2/faq.html#noconfig for more info.
```

This can safely be ignored and will be addressed in a future patch.
