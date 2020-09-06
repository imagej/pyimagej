# Common Errors

## I ran a plugin and see an updated image, but the numpy array and dataset are unchanged.
This bug can occur in certain circumstances when using ImageJ1 plugins which update a corresponding ImagePlus.
It can be worked around by calling
```python
import imagej
ij = imagej.init(['net.imagej:imagej', 'net.imagej:imagej-legacy'])
imp = ij.py.get_img_plus()
```
Once you have `imp`, the `ImagePlus`, you can call subsequently use the `synchronize_ij1_to_ij2` function to
update it.
```python
ij.py.synchronize_ij1_to_ij2(imp)
```

## The GUI does not work on mac
The default UI does not currently work on mac.  You can set up a Swing UI based GUI instead by initializing
with:

```python
import imagej
ij = imagej.init('/Applications/Fiji.app', headless=False)
ij.ui().showUI("swing")
```

See [this thread](https://github.com/imagej/pyimagej/issues/23) for additional information and updates.

## Tab completion crashes the kernel
This is a known error.  While we do not have a direct solution, see [this thread](https://github.com/imagej/pyimagej/issues/34) for a work around

## Pyjnius 1.2.1 incompatibility
Version 1.2.1 of pyjnius causes several bugs with pyimagej.  These bugs can be fixed by using
version 1.2.0.  These bugs include

1. `libjvm.so not found`
2. `AttributeError: 'some.class' object has no attribute 'function'`


## NameError: name 'ij' is not defined
This error occurs when you import jnius before imagej.init().  Both use the Java Virtual Machine (JVM), so initializing 
jnius prevents imagej.init from starting up.

## ImageJ1 classes not found
If you try to load an ImageJ1 class of path `ij.X`, and get a `JavaException: Class not found`
error, this is because ImageJ was initialized without ImageJ1.  See [Initialization.md](Initialization.md)

## Not enough memory
You can increase the memory available to the JVM before starting ImageJ.  See [Initialization.md](Initialization.md)

## log4j:WARN 
PyImageJ does not currently ship a log4j implementation, which results in an obnoxious warning at startup:

```
log4j:WARN No appenders could be found for logger (org.bushe.swing.event.EventService).
log4j:WARN Please initialize the log4j system properly.
log4j:WARN See http://logging.apache.org/log4j/1.2/faq.html#noconfig for more info.
```

This can safely be ignored and will be addressed in a future patch.
