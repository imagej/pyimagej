# Common Errors

## I ran a plugin and see an updated image, but the numpy array and dataset are unchanged.
This bug occurs when using ImageJ1 plugins which update a corresponding ImagePlus.
It can be worked around by calling
```python
import imagej
ij = imagej.init('net.imagej:imagej+net.imagej:imagej-legacy')
imp = ij.py.get_img_plus()
# For a single modification, and after subsequent modifications calling
ij.py.synchronize_ij1_to_ij2(imp)
```

## The GUI does not work on mac
There are no current solutions.  See [this thread](issues/23) for updates.

## Tab completion crashes the kernel
This is a known error.  While we do not have a direct solution, you can call `dir(x)`
where x is whatever object you were tab completing off of to get a list of functions.  For example.
```python
import imagej
ij = imagej.init()
funcs = dir(ij.ui())
for item in funcs:
    if not item.startswith('_'):  # Ignore private functions
        print(item)
```

## Pyjnius 1.2.1 incompatibility
Version 1.2.1 of pyjnius causes several bugs with pyimagej.  These bugs can be fixed by using
version 1.2.0.

## Importing jnius before imagej.init()
You cannot import jnius before ImageJ.  Both use the Java Virtual Machine (JVM), so initializing jnius prevents imagej.init from starting up.

## ImageJ1 classes not found
If you try to load an ImageJ1 class of path `ij.X`, and get a `JavaException: Class not found`
error, this is because ImageJ was initialized without ImageJ1.  See [INITIALIZATION.md](Initialization.md)

## Not enough memory
You can increase the memory available to the JVM before starting ImageJ.  See [INITIALIZATION.md](Initialization.md)

## log4j:WARN 
PyImageJ does not currently ship a log4j implementation, which results in an obnoxious warning at startup:

```
log4j:WARN No appenders could be found for logger (org.bushe.swing.event.EventService).
log4j:WARN Please initialize the log4j system properly.
log4j:WARN See http://logging.apache.org/log4j/1.2/faq.html#noconfig for more info.
```

This can safely be ignored and will be addressed in a future patch.