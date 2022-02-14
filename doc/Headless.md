# Using ImageJ in headless mode

It is an increasingly common scenario to want to do image processing on a cloud
computing node (e.g. running notebooks on [Binder](https://mybinder.org/) or
[Google Colab](https://colab.research.google.com)). Unfortunately, the
original ImageJ was only designed to be a GUI-based desktop application, so it
does not natively support true
[headless](https://en.wikipedia.org/wiki/Headless_computer) operation, i.e.
without a display attached.

The [ImageJ2](https://imagej.net/software/imagej2) project, however,
does support headless operation for all its functions, due to its careful
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
[Xvfb](Xvfb) to run ImageJ inside a "virtual" graphical environment without a
physical screen present.

## Starting PyImageJ in headless mode

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

## Troubleshooting

See the
[Known Limitations section of the Troubleshooting guide](Troubleshooting#known-limitations)
for some further details about what does and does not work headless, and
things to try when having difficulty with ImageJ's behavior in headless mode.
