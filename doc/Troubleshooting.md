# Common Errors

## Error in "mvn.CMD -B -f pom.xml" dependency:resolve: 1

This indicates a problem running Maven on your system and will require more
debugging effort. Please post [on the
forum](https://forum.image.sc/tag/pyimagej) and include either:

* The results of manually running the Maven command with an added `-X` flag: `path\to\mvn.CMD -B -f -X path\to\pom.xml`
* The results of re-running the same `imagej.init` call after:
   * Deleting your `~/.jgo` directory
   * Adding `import logging` and `logging.basicConfig(level = logging.DEBUG)` to the top of your script

## I ran a plugin and see an updated image, but the numpy array and dataset are unchanged.

This bug can occur in certain circumstances when using original ImageJ plugins
which update a corresponding `ImagePlus`. It can be worked around by calling:

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

## Original ImageJ classes not found

If you try to load an original ImageJ class (with package prefix `ij`),
and get a `JavaException: Class not found` error, this is because
the environment was initialized without the original ImageJ included.
See [Initialization.md](Initialization.md).

## Not enough memory

You can increase the memory available to the JVM before starting it.
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
